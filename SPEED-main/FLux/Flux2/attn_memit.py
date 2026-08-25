import os, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import argparse
import pandas as pd
from diffusers.utils import logging as diffusers_logging
from safetensors.torch import save_file
from tqdm import tqdm
from diffusers import Flux2KleinPipeline


ATTENTION_SUFFIXES = {"Q": ".attn.add_q_proj", "K": ".attn.add_k_proj", "V": ".attn.add_v_proj"}

def _trace_concepts(
    pipeline,
    concepts,
    token_indices,
    module_names,
    args,
    device,
    max_sequence_length,
    progress_bar=None,
):
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
            traces = {name: {"inputs": [], "outputs": []} for name in module_names}
            handles = []
            for name in module_names:
                module = module_lookup[name]

                def pre_hook(_module, inputs, module_name=name):
                    selected_inputs = inputs[0][:, selected_token_indices, :]
                    traces[module_name]["inputs"].append(selected_inputs.detach().float())

                def out_hook(_module, _inputs, output, module_name=name):
                    output = output[0] if isinstance(output, tuple) else output
                    output = output[:, selected_token_indices, :]
                    traces[module_name]["outputs"].append(output.detach().float())

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(out_hook))

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
                    output_steps = torch.stack(record["outputs"], dim=0)
                    compact[name] = {
                        "inputs": input_steps[:, batch_index, :, :].reshape(-1, input_steps.shape[-1]).T,
                        "outputs": output_steps[:, batch_index, :, :].reshape(-1, output_steps.shape[-1]).T,
                    }
                traced_concepts[concept] = compact
            if progress_bar is not None:
                progress_bar.update(len(concept_batch))
    return traced_concepts

def _closed_form_update(residual_target, target_target, update_lambda, retain_inputs, retain_threshold):
    retain_inputs = retain_inputs.to(device=target_target.device, dtype=target_target.dtype)
    covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
    null_basis = U[:, S < retain_threshold]
    eye = torch.eye(target_target.shape[0], device=target_target.device, dtype=target_target.dtype)
    projector = eye if null_basis.shape[1] == 0 else null_basis @ null_basis.T
    system = target_target @ projector + update_lambda * eye
    return torch.linalg.solve(system.T, (residual_target @ projector).T).T


def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, device="cuda:0", max_sequence_length=512):
    selected_params = list(args.params)
    edit_modules = []
    for name, module in pipeline.transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        match = re.match(r"transformer_blocks\.(\d+)\.", name)
        if match is None:
            continue
        matched_params = [
            param
            for param in selected_params
            if name.endswith(ATTENTION_SUFFIXES[param])
        ]
        if matched_params:
            edit_modules.append((name, module, matched_params[0]))
    if not edit_modules:
        raise RuntimeError(f"No text-side attention modules selected for params={args.params}")
    module_names = [name for name, _, _ in edit_modules]
    grouped_modules = {}
    modules_by_param = {param: [] for param in selected_params}
    for module_name, module, param in edit_modules:
        layer_index = int(re.match(r"transformer_blocks\.(\d+)\.", module_name).group(1))
        grouped_modules.setdefault(layer_index, []).append((module_name, module, param))
        modules_by_param[param].append((module_name, module))
    final_module_by_param = {
        param: modules[-1][0]
        for param, modules in modules_by_param.items()
        if modules
    }
    remaining_counts_by_param = {
        param: {
            module_name: len(modules) - index
            for index, (module_name, _module) in enumerate(modules)
        }
        for param, modules in modules_by_param.items()
        if modules
    }

    non_empty_concepts = [
        concept
        for concept in dict.fromkeys(target_concepts + anchor_concepts + retain_texts)
        if concept != ""
    ]
    concept_token_indices = {}
    for concept in non_empty_concepts:
        text = pipeline.tokenizer.apply_chat_template(
            [{"role": "user", "content": concept}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        suffix_text = text.split(concept, 1)[1]
        suffix_length = int(pipeline.tokenizer(
            suffix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).attention_mask[0].sum().item())
        full_length = int(pipeline.tokenizer(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).attention_mask[0].sum().item())
        token_index = full_length - suffix_length - 1
        if token_index < 0:
            raise RuntimeError(f"Prompt token for {concept!r} was truncated by max_sequence_length={max_sequence_length}.")
        concept_token_indices[concept] = [token_index]

    target_token_indices = {
        concept: concept_token_indices[concept]
        for concept in target_concepts
    }
    anchor_token_indices = {
        concept: [0] if concept == "" else concept_token_indices[concept]
        for concept in anchor_concepts
    }
    retain_token_indices = {
        concept: list(range(1, max_sequence_length)) if concept == "" else concept_token_indices[concept]
        for concept in retain_texts
    }

    anchor_final_traces = _trace_concepts(
        pipeline,
        anchor_concepts,
        anchor_token_indices,
        list(final_module_by_param.values()),
        args,
        device,
        max_sequence_length,
    )

    retain_inputs_by_module = {module_name: [] for module_name in module_names}
    retain_total = sum(
        1
        for concept in retain_texts
        if retain_token_indices.get(concept)
    )
    with tqdm(total=retain_total, desc="retain trace", dynamic_ncols=True, leave=True) as retain_pbar:
        for j in range(0, len(retain_texts), args.chunk_size):
            retain_chunk = retain_texts[j:j + args.chunk_size]
            retain_traces = _trace_concepts(
                pipeline,
                retain_chunk,
                retain_token_indices,
                module_names,
                args,
                device,
                max_sequence_length,
                progress_bar=retain_pbar,
            )
            for module_name in module_names:
                retain_inputs = [
                    retain_traces[concept][module_name]["inputs"]
                    for concept in retain_chunk
                    if concept in retain_traces and module_name in retain_traces[concept]
                ]
                if retain_inputs:
                    retain_inputs_by_module[module_name].append(torch.cat(retain_inputs, dim=1))
            del retain_traces
    for module_name in module_names:
        if retain_inputs_by_module[module_name]:
            retain_inputs_by_module[module_name] = torch.cat(retain_inputs_by_module[module_name], dim=1)
        else:
            retain_inputs_by_module[module_name] = None

    edit_dict = {}
    for _layer_index, layer_modules in sorted(grouped_modules.items()):
        layer_total = len(target_concepts) + len(layer_modules)
        with tqdm(total=layer_total, desc=f"layer {_layer_index}", dynamic_ncols=True, leave=False) as layer_pbar:
            for module_name, module, param in layer_modules:
                final_module_name = final_module_by_param[param]
                trace_names = [module_name] if module_name == final_module_name else [module_name, final_module_name]
                target_traces = _trace_concepts(
                    pipeline,
                    target_concepts,
                    target_token_indices,
                    trace_names,
                    args,
                    device,
                    max_sequence_length,
                )
                target_inputs, residuals = [], []
                for concept, anchor_concept in zip(target_concepts, anchor_concepts):
                    concept_trace = target_traces[concept]
                    current_final = concept_trace[final_module_name]["outputs"]
                    anchor_final = anchor_final_traces[anchor_concept][final_module_name]["outputs"].to(current_final.device, current_final.dtype)
                    target_inputs.append(concept_trace[module_name]["inputs"])
                    residuals.append(
                        args.residual_scale
                        * (anchor_final - current_final)
                        / remaining_counts_by_param[param][module_name]
                    )

                target_target = torch.stack([target @ target.T for target in target_inputs]).mean(0)
                residual_target = torch.stack([residual @ target.T for target, residual in zip(target_inputs, residuals)]).mean(0)
                residual_target = residual_target.to(module.weight.device, torch.float32)
                target_target = target_target.to(module.weight.device, torch.float32)
                retain_inputs = retain_inputs_by_module[module_name]

                delta = _closed_form_update(
                    residual_target,
                    target_target,
                    args.update_lambda,
                    None if retain_inputs is None else retain_inputs.to(module.weight.device, torch.float32),
                    args.threshold,
                )
                module.weight = torch.nn.Parameter(module.weight.float().add(delta).to(module.weight.dtype))
                edit_dict[module_name + ".weight"] = module.weight.detach().clone()
                layer_pbar.update(1)
    return edit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", help="base version for FLUX", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--trace_batch_size", type=int, default=4)
    parser.add_argument("--params", type=str, default="KV", choices=["Q", "K", "V", "QK", "KV", "QKV"])
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--trace_num_steps", type=int, default=4)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    args = parser.parse_args()

    diffusers_logging.set_verbosity_error()
    try:
        diffusers_logging.disable_progress_bar()
    except AttributeError:
        pass

    target_concepts = [con.strip() for con in args.target_concepts.split(",")]
    if not target_concepts or any(concept == "" for concept in target_concepts):
        raise ValueError("--target_concepts must not contain empty concepts")
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path

    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}-attn-memit"
    anchor_concepts = [x.strip() for x in anchor_concepts.split(",")]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
        if anchor_concepts[0] == "":
            file_suffix += "-to_null"
        else:
            file_suffix += f"-to_{anchor_concepts[0]}"
    else:
        assert len(target_concepts) == len(anchor_concepts)
        file_suffix += f"-to_{anchor_concepts[0]}_etc"

    retain_texts = []
    if retain_path is not None:
        assert retain_path.endswith(".csv")
        df = pd.read_csv(retain_path)
        for head in args.heads.split(","):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")
    retain_texts = [
        text
        for text in retain_texts
        if not any(re.search(r"\b" + re.escape(concept.lower()) + r"\b", str(text).lower()) for concept in target_concepts)
    ]

    pipeline = Flux2KleinPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    try:
        pipeline.set_progress_bar_config(disable=True)
    except AttributeError:
        pass
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
