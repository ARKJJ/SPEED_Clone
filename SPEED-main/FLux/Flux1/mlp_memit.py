import os, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import argparse
import pandas as pd
from safetensors.torch import save_file
from diffusers import DiffusionPipeline

FLUX1_MLP_SUFFIX = ".ff_context.net.2"
FLUX1_MAX_SEQUENCE_LENGTH = 256


def _trace_concepts(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):
    module_lookup = dict(pipeline.transformer.named_modules())
    traces_by_concept = {}
    groups = {}
    for concept in dict.fromkeys(concepts):
        indices = token_indices.get(concept)
        if indices:
            groups.setdefault(tuple(indices), []).append(concept)

    for indices, concepts_with_indices in groups.items():
        indices = list(indices)
        for start in range(0, len(concepts_with_indices), args.trace_batch_size):
            concept_batch = concepts_with_indices[start:start + args.trace_batch_size]
            traces = {name: {"inputs": [], "outputs": []} for name in module_names}
            handles = []
            for name in module_names:
                module = module_lookup[name]

                def pre_hook(_module, inputs, module_name=name):
                    values = inputs[0][:, indices, :]
                    traces[module_name]["inputs"].append(values.detach().float())

                def output_hook(_module, _inputs, output, module_name=name):
                    values = output[0] if isinstance(output, tuple) else output
                    values = values[:, indices, :]
                    traces[module_name]["outputs"].append(values.detach().float())

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(output_hook))

            generators = [torch.Generator(device=device).manual_seed(args.trace_seed) for _ in concept_batch]
            with torch.no_grad():
                pipeline(
                    prompt=concept_batch,
                    generator=generators,
                    num_inference_steps=args.trace_num_steps,
                    guidance_scale=args.trace_guidance_scale,
                    height=args.trace_resolution,
                    width=args.trace_resolution,
                    max_sequence_length=max_sequence_length,
                    output_type="latent",
                )
            for handle in handles:
                handle.remove()
            for batch_index, concept in enumerate(concept_batch):
                traces_by_concept[concept] = {
                    name: {
                        kind: torch.stack(record[kind], dim=0)[:, batch_index, :, :].reshape(
                            -1, torch.stack(record[kind], dim=0).shape[-1]
                        ).T
                        for kind in ("inputs", "outputs")
                    }
                    for name, record in traces.items()
                }
    return traces_by_concept


def _closed_form_update(residual_target, target_target, update_lambda, retain_inputs, retain_threshold):
    retain_inputs = retain_inputs.to(device=target_target.device, dtype=target_target.dtype)
    covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
    null_basis = U[:, S < retain_threshold]
    eye = torch.eye(target_target.shape[0], device=target_target.device, dtype=target_target.dtype)
    projector = eye if null_basis.shape[1] == 0 else null_basis @ null_basis.T
    system = target_target @ projector + update_lambda * eye
    return torch.linalg.solve(system.T, (residual_target @ projector).T).T


def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, device="cuda:0", max_sequence_length=FLUX1_MAX_SEQUENCE_LENGTH):
    edit_modules = []
    for name, module in pipeline.transformer.named_modules():
        match = re.match(r"transformer_blocks\.(\d+)\.", name)
        if match is None or not name.endswith(FLUX1_MLP_SUFFIX) or not hasattr(module, "weight") or module.weight is None:
            continue
        edit_modules.append((name, module))
    if not edit_modules:
        raise RuntimeError("No Flux1 text MLP modules found: expected transformer_blocks.*.ff_context.net.2")
    module_names = [name for name, _ in edit_modules]
    final_module_name = module_names[-1]
    remaining_counts = {name: len(module_names) - index for index, name in enumerate(module_names)}

    non_empty_concepts = [
        concept for concept in dict.fromkeys(target_concepts + anchor_concepts + retain_texts)
        if concept != ""
    ]
    concept_token_indices = {}
    for concept in non_empty_concepts:
        token_inputs = pipeline.tokenizer_2(
            concept,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        valid_token_count = int(token_inputs.attention_mask[0].sum().item())
        content_indices = list(range(valid_token_count - 1))
        if not content_indices:
            raise RuntimeError(f"No content token found for {concept!r}.")
        concept_token_indices[concept] = content_indices

    target_token_indices = {concept: concept_token_indices[concept] for concept in target_concepts}
    anchor_token_indices = {
        concept: [0] if concept == "" else [concept_token_indices[concept][-1]]
        for concept in anchor_concepts
    }
    retain_token_indices = {
        concept: list(range(1, max_sequence_length)) if concept == "" else [concept_token_indices[concept][-1]]
        for concept in retain_texts
    }

    anchor_final_traces = _trace_concepts(
        pipeline, anchor_concepts, anchor_token_indices, [final_module_name], args, device, max_sequence_length
    )
    retain_inputs_by_module = {module_name: [] for module_name in module_names}
    for start in range(0, len(retain_texts), args.chunk_size):
        retain_chunk = retain_texts[start:start + args.chunk_size]
        retain_traces = _trace_concepts(
            pipeline, retain_chunk, retain_token_indices, module_names, args, device, max_sequence_length
        )
        for module_name in module_names:
            inputs = [
                retain_traces[concept][module_name]["inputs"]
                for concept in retain_chunk
                if concept in retain_traces and module_name in retain_traces[concept]
            ]
            if inputs:
                retain_inputs_by_module[module_name].append(torch.cat(inputs, dim=1))
        del retain_traces
    for module_name in module_names:
        if not retain_inputs_by_module[module_name]:
            raise RuntimeError(f"No retain trace for {module_name}")
        retain_inputs_by_module[module_name] = torch.cat(retain_inputs_by_module[module_name], dim=1)

    edit_dict = {}
    for module_name, module in edit_modules:
        trace_names = [module_name] if module_name == final_module_name else [module_name, final_module_name]
        target_traces = _trace_concepts(
            pipeline, target_concepts, target_token_indices, trace_names, args, device, max_sequence_length
        )
        target_inputs, residuals = [], []
        for concept, anchor_concept in zip(target_concepts, anchor_concepts):
            current_final = target_traces[concept][final_module_name]["outputs"]
            anchor_final = anchor_final_traces[anchor_concept][final_module_name]["outputs"].to(
                current_final.device, current_final.dtype
            )
            if current_final.shape[1] % anchor_final.shape[1] != 0:
                raise RuntimeError(
                    f"Trace shape mismatch: target={current_final.shape}, "
                    f"anchor={anchor_final.shape}"
                )
            target_count = current_final.shape[1] // anchor_final.shape[1]
            anchor_final = anchor_final.repeat_interleave(target_count, dim=1)
            target_inputs.append(target_traces[concept][module_name]["inputs"])
            residuals.append(
                args.residual_scale * (anchor_final - current_final) / remaining_counts[module_name]
            )

        target_target = torch.stack([
            target @ target.T / target.shape[1] for target in target_inputs
        ]).mean(0)
        residual_target = torch.stack([
            residual @ target.T / target.shape[1]
            for target, residual in zip(target_inputs, residuals)
        ]).mean(0)
        delta = _closed_form_update(
            residual_target.to(module.weight.device, torch.float32),
            target_target.to(module.weight.device, torch.float32),
            args.update_lambda,
            retain_inputs_by_module[module_name].to(module.weight.device, torch.float32),
            args.threshold,
        )
        module.weight = torch.nn.Parameter(module.weight.float().add(delta).to(module.weight.dtype))
        edit_dict[module_name + ".weight"] = module.weight.detach().clone()
    return edit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--trace_batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=3e-2)
    parser.add_argument("--trace_num_steps", type=int, default=20)
    parser.add_argument("--trace_guidance_scale", type=float, default=3.5)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--max_sequence_length", type=int, default=FLUX1_MAX_SEQUENCE_LENGTH)
    args = parser.parse_args()

    target_concepts = [con.strip() for con in args.target_concepts.split(",")]
    anchor_concepts = [con.strip() for con in args.anchor_concepts.split(",")]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
    else:
        assert len(target_concepts) == len(anchor_concepts)

    retain_texts = []
    if args.retain_path is not None:
        assert args.retain_path.endswith(".csv")
        dataframe = pd.read_csv(args.retain_path)
        for head in args.heads.split(","):
            retain_texts += dataframe[head.strip()].unique().tolist()
    else:
        retain_texts.append("")
    retain_texts = [
        text for text in retain_texts
        if not any(re.search(r"\b" + re.escape(concept.lower()) + r"\b", text.lower()) for concept in target_concepts)
    ]

    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}-mlp-memit"
    file_suffix += "-to_null" if anchor_concepts[0] == "" else f"-to_{anchor_concepts[0]}"
    pipeline = DiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    edit_dict = edit_model(
        args, pipeline, target_concepts, anchor_concepts, retain_texts, args.device, args.max_sequence_length
    )
    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    save_file(edit_dict, os.path.join(save_path, f"{file_name}.safetensors"))
