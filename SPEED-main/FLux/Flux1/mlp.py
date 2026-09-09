import os, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import pandas as pd
from safetensors.torch import save_file
from diffusers import DiffusionPipeline

FLUX1_MLP_SUFFIX = ".ff_context.net.2"


def _add_matrix(total, matrix):
    return matrix if total is None else total.add_(matrix)


def _trace_concepts(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length, on_concept_trace=None):
    module_lookup = dict(pipeline.transformer.named_modules())
    traced_concepts = {}
    trace_batch_size = max(1, int(getattr(args, "trace_batch_size", 1)))
    grouped_concepts = {}

    for concept in dict.fromkeys(concepts):
        selected_token_indices = token_indices.get(concept)
        if not selected_token_indices:
            continue
        grouped_concepts.setdefault(tuple(selected_token_indices), []).append(concept)

    for selected_token_indices, grouped in grouped_concepts.items():
        selected_token_indices = list(selected_token_indices)
        for start in range(0, len(grouped), trace_batch_size):
            concept_batch = grouped[start:start + trace_batch_size]
            traces = {name: {"inputs": []} for name in module_names}
            handles = []
            for name in module_names:
                module = module_lookup[name]

                def pre_hook(_module, inputs, module_name=name):
                    selected_inputs = inputs[0][:, selected_token_indices, :]
                    traces[module_name]["inputs"].append(selected_inputs.detach().float())

                handles.append(module.register_forward_pre_hook(pre_hook))

            generators = [
                torch.Generator(device=device).manual_seed(args.trace_seed)
                for _ in concept_batch
            ]
            with torch.no_grad():
                pipeline(
                    prompt=concept_batch,
                    generator=generators,
                    num_inference_steps=args.trace_num_steps,
                    guidance_scale=3.5,
                    height=args.trace_resolution,
                    width=args.trace_resolution,
                    max_sequence_length=max_sequence_length,
                    output_type="latent",
                )
            for handle in handles:
                handle.remove()

            for batch_index, concept in enumerate(concept_batch):
                compact = {}
                for name, record in traces.items():
                    input_steps = torch.stack(record["inputs"], dim=0)
                    compact[name] = {
                        "inputs": input_steps[:, batch_index, :, :].reshape(-1, input_steps.shape[-1]).T,
                    }
                if on_concept_trace is None:
                    traced_concepts[concept] = compact
                else:
                    on_concept_trace(concept, compact)
                    del compact
    return traced_concepts


def _closed_form_update(sum_target_anchor, sum_target_target, weight, update_lambda, retain_second_moment, retain_count, retain_threshold=1e-1):
    covariance = retain_second_moment.to(device=sum_target_target.device, dtype=sum_target_target.dtype) / retain_count
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
    null_basis = U[:, S < retain_threshold]
    if null_basis.shape[1] == 0:
        projector = torch.eye(sum_target_target.shape[0], device=sum_target_target.device, dtype=sum_target_target.dtype)
    else:
        projector = null_basis @ null_basis.T
    eye = torch.eye(sum_target_target.shape[0], device=sum_target_target.device, dtype=sum_target_target.dtype)
    system = sum_target_target @ projector + update_lambda * eye
    residual_projection = weight @ (sum_target_anchor - sum_target_target) @ projector
    delta = torch.linalg.solve(system.T, residual_projection.T).T
    return delta


def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, device="cuda:0", max_sequence_length=256,):
    edit_modules = []
    for name, module in pipeline.transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not name.endswith(FLUX1_MLP_SUFFIX):
            continue
        if re.match(r"transformer_blocks\.(\d+)\.", name) is None:
            continue
        edit_modules.append((name, module))
    if not edit_modules:
        raise RuntimeError("No Flux1 text MLP modules found: expected transformer_blocks.*.ff_context.net.2")
    module_names = [name for name, _ in edit_modules]
    grouped_modules = {}
    for module_name, module in edit_modules:
        match = re.match(r"transformer_blocks\.(\d+)\.", module_name)
        if match is None:
            continue
        grouped_modules.setdefault(int(match.group(1)), []).append((module_name, module))

    non_empty_concepts = [concept for concept in dict.fromkeys(target_concepts + anchor_concepts + retain_texts) if concept]
    concept_token_indices = {}
    all_token_indices = {}
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
            raise RuntimeError(f"Prompt token for {concept!r} was truncated by max_sequence_length={max_sequence_length}.")
        concept_token_indices[concept] = content_indices
        all_token_indices[concept] = list(range(valid_token_count))

    target_token_indices = {concept: all_token_indices[concept] for concept in target_concepts}
    anchor_token_indices = {
        concept: [0] if concept == "" else [concept_token_indices[concept][-1]]
        for concept in anchor_concepts
    }
    retain_token_indices = {
        concept: list(range(1, max_sequence_length)) if concept == "" else [concept_token_indices[concept][-1]]
        for concept in retain_texts
    }

    anchor_base_traces = _trace_concepts(
        pipeline,
        anchor_concepts,
        anchor_token_indices,
        module_names,
        args,
        device,
        max_sequence_length,
    )

    retain_second_moment_by_module = {module_name: None for module_name in module_names}
    retain_count_by_module = {module_name: 0 for module_name in module_names}

    def accumulate_retain_trace(_concept, concept_trace):
        for module_name in module_names:
            retain_inputs = concept_trace[module_name]["inputs"]
            retain_second_moment_by_module[module_name] = _add_matrix(
                retain_second_moment_by_module[module_name], retain_inputs @ retain_inputs.T
            )
            retain_count_by_module[module_name] += retain_inputs.shape[1]

    for j in range(0, len(retain_texts), args.chunk_size):
        retain_chunk = retain_texts[j:j + args.chunk_size]
        _trace_concepts(
            pipeline,
            retain_chunk,
            retain_token_indices,
            module_names,
            args,
            device,
            max_sequence_length,
            on_concept_trace=accumulate_retain_trace,
        )
    for module_name in module_names:
        if retain_second_moment_by_module[module_name] is None:
            raise RuntimeError(f"No retain trace for {module_name}")

    edit_dict = {}
    for _layer_index, layer_modules in sorted(grouped_modules.items()):
        layer_module_names = [module_name for module_name, _module in layer_modules]
        target_to_anchor = dict(zip(target_concepts, anchor_concepts))
        target_second_moment_by_module = {module_name: None for module_name in layer_module_names}
        target_cross_moment_by_module = {module_name: None for module_name in layer_module_names}
        target_matrix_count_by_module = {module_name: 0 for module_name in layer_module_names}

        def accumulate_target_trace(concept, concept_trace):
            anchor_concept = target_to_anchor[concept]
            for module_name in layer_module_names:
                target_inputs = concept_trace[module_name]["inputs"]
                anchor_inputs = anchor_base_traces[anchor_concept][module_name]["inputs"]
                if target_inputs.shape[1] % anchor_inputs.shape[1] != 0:
                    raise RuntimeError(
                        f"Trace shape mismatch: target={target_inputs.shape}, anchor={anchor_inputs.shape}"
                    )
                target_count = target_inputs.shape[1] // anchor_inputs.shape[1]
                anchor_inputs = anchor_inputs.repeat_interleave(target_count, dim=1)
                target_second_moment_by_module[module_name] = _add_matrix(
                    target_second_moment_by_module[module_name], target_inputs @ target_inputs.T
                )
                target_cross_moment_by_module[module_name] = _add_matrix(
                    target_cross_moment_by_module[module_name], anchor_inputs @ target_inputs.T
                )
                target_matrix_count_by_module[module_name] += 1

        _trace_concepts(
            pipeline,
            target_concepts,
            target_token_indices,
            layer_module_names,
            args,
            device,
            max_sequence_length,
            on_concept_trace=accumulate_target_trace,
        )
        for module_name, module in layer_modules:
            if target_second_moment_by_module[module_name] is None:
                raise RuntimeError(f"No target trace for {module_name}")
            matrix_count = target_matrix_count_by_module[module_name]
            sum_target_target = target_second_moment_by_module[module_name] / matrix_count
            sum_target_anchor = target_cross_moment_by_module[module_name] / matrix_count
            sum_target_anchor = sum_target_anchor.to(module.weight.device, torch.float32)
            sum_target_target = sum_target_target.to(module.weight.device, torch.float32)
            weight_before = module.weight.float()
            delta = _closed_form_update(
                sum_target_anchor,
                sum_target_target,
                weight_before,
                args.update_lambda,
                retain_second_moment_by_module[module_name],
                retain_count_by_module[module_name],
                args.threshold,
            )
            module.weight = torch.nn.Parameter(weight_before.add(delta).to(module.weight.dtype))
            edit_dict[module_name + ".weight"] = module.weight.detach().clone()

    return edit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", help="base version for FLUX", type=str, default="black-forest-labs/FLUX.1-dev")
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
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=0.1)
    args = parser.parse_args()

    target_concepts = [con.strip() for con in args.target_concepts.split(",")]
    if not target_concepts or any(concept == "" for concept in target_concepts):
        raise ValueError("--target_concepts must not contain empty concepts")
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path

    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}-mlp2"
    anchor_concepts = [x.strip() for x in anchor_concepts.split(",")]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
        if anchor_concepts[0] == "":
            file_suffix += "-to_null"
        else:
            file_suffix += f"-to_{anchor_concepts[0]}"
    else:
        assert len(target_concepts) == len(anchor_concepts)
        file_suffix += f'-to_{anchor_concepts[0]}_etc'

    retain_texts = []
    if retain_path is not None:
        assert retain_path.endswith('.csv')
        df = pd.read_csv(retain_path)
        for head in args.heads.split(','):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")
    retain_texts = [
        text for text in retain_texts
        if not any(re.search(r"\b" + re.escape(concept.lower()) + r"\b", text.lower()) for concept in target_concepts)
    ]

    pipeline = DiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    edit_dict = edit_model(
        args=args,
        pipeline=pipeline,
        target_concepts=target_concepts,
        anchor_concepts=anchor_concepts,
        retain_texts=retain_texts,
        device=args.device,
        max_sequence_length=512,
    )

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    save_file(edit_dict, os.path.join(save_path, f"{file_name}.safetensors"))
