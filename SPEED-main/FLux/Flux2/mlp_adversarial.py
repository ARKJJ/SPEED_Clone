import argparse
import os
import re
import time

import pandas as pd
import torch
from diffusers import Flux2KleinPipeline
from safetensors.torch import load_file, save_file


FLUX2_MLP_SUFFIX = ".ff_context.linear_out"


def _trace_concepts(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):
    module_lookup = dict(pipeline.transformer.named_modules())
    traced_concepts = {}
    grouped_concepts = {}
    for concept in dict.fromkeys(concepts):
        selected_token_indices = token_indices.get(concept)
        if selected_token_indices:
            grouped_concepts.setdefault(tuple(selected_token_indices), []).append(concept)

    for selected_token_indices, grouped in grouped_concepts.items():
        for start in range(0, len(grouped), max(1, args.trace_batch_size)):
            concept_batch = grouped[start:start + max(1, args.trace_batch_size)]
            traces = {name: [] for name in module_names}
            handles = []
            for name in module_names:
                def pre_hook(_module, inputs, module_name=name):
                    values = inputs[0][:, list(selected_token_indices), :]
                    traces[module_name].append(values.detach().float())

                handles.append(module_lookup[name].register_forward_pre_hook(pre_hook))

            generators = [torch.Generator(device=device).manual_seed(args.trace_seed) for _ in concept_batch]
            with torch.no_grad():
                pipeline(
                    prompt=concept_batch,
                    generator=generators,
                    num_inference_steps=args.trace_num_steps,
                    height=args.trace_resolution,
                    width=args.trace_resolution,
                    max_sequence_length=max_sequence_length,
                    output_type="latent",
                )
            for handle in handles:
                handle.remove()

            for batch_index, concept in enumerate(concept_batch):
                traced_concepts[concept] = {
                    name: torch.stack(values, dim=0)[:, batch_index].reshape(-1, values[0].shape[-1]).T
                    for name, values in traces.items()
                }
    return traced_concepts


def _adversarial_inputs(inputs, base_weight, current_weight, adversarial_lambda):
    inputs = inputs.to(device=current_weight.device, dtype=torch.float32)
    base_weight = base_weight.to(device=current_weight.device, dtype=torch.float32)
    current_weight = current_weight.to(dtype=torch.float32)
    eye = torch.eye(current_weight.shape[1], device=current_weight.device, dtype=current_weight.dtype)
    system = adversarial_lambda * eye + current_weight.T @ current_weight
    right_hand_side = current_weight.T @ (base_weight @ inputs)
    return torch.linalg.solve(system, right_hand_side)


def _closed_form_update(sum_target_anchor, sum_target_target, weight, update_lambda, retain_inputs, retain_threshold=1e-1):
    retain_inputs = retain_inputs.to(device=sum_target_target.device, dtype=sum_target_target.dtype)
    covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
    null_basis = U[:, S < retain_threshold]
    if null_basis.shape[1] == 0:
        projector = torch.eye(sum_target_target.shape[0], device=sum_target_target.device, dtype=sum_target_target.dtype)
    else:
        projector = null_basis @ null_basis.T
    eye = torch.eye(sum_target_target.shape[0], device=sum_target_target.device, dtype=sum_target_target.dtype)
    system = sum_target_target @ projector + update_lambda * eye
    residual_projection = weight @ (sum_target_anchor - sum_target_target) @ projector
    return torch.linalg.solve(system.T, residual_projection.T).T


def _select_edit_modules(pipeline):
    edit_modules = []
    for name, module in pipeline.transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not name.endswith(FLUX2_MLP_SUFFIX):
            continue
        if re.match(r"transformer_blocks\.(\d+)\.", name) is None:
            continue
        edit_modules.append((name, module))
    if not edit_modules:
        raise RuntimeError("No Flux2 text MLP modules found: expected transformer_blocks.*.ff_context.linear_out")
    return edit_modules


def _load_sparse_weights(pipeline, checkpoint_path):
    state_dict = load_file(checkpoint_path, device="cpu")
    transformer_state = pipeline.transformer.state_dict()
    for key, value in state_dict.items():
        state_key = key[len("transformer."):] if key.startswith("transformer.") else key
        if state_key not in transformer_state:
            raise KeyError(f"Edited weight {key!r} is not in the FLUX transformer state dict")
        transformer_state[state_key].copy_(value.to(transformer_state[state_key].device, transformer_state[state_key].dtype))
    print(f"Loaded {len(state_dict)} edited weights from {checkpoint_path}")


def _collect_retain_inputs(pipeline, retain_texts, module_names, args, device, max_sequence_length):
    """Trace retain activations once on the untouched base model and keep them fixed."""
    full_token_indices = list(range(max_sequence_length))
    retain_token_indices = {text: full_token_indices for text in retain_texts}
    retain_inputs_by_module = {module_name: [] for module_name in module_names}
    for start in range(0, len(retain_texts), args.chunk_size):
        retain_chunk = retain_texts[start:start + args.chunk_size]
        retain_traces = _trace_concepts(
            pipeline,
            retain_chunk,
            retain_token_indices,
            module_names,
            args,
            device,
            max_sequence_length,
        )
        for module_name in module_names:
            inputs = [
                retain_traces[text][module_name]
                for text in retain_chunk
                if text in retain_traces
            ]
            if inputs:
                retain_inputs_by_module[module_name].append(torch.cat(inputs, dim=1).cpu())
    fixed_retain_inputs = {}
    for module_name, chunks in retain_inputs_by_module.items():
        if not chunks:
            raise RuntimeError(f"No retain trace for {module_name}")
        fixed_retain_inputs[module_name] = torch.cat(chunks, dim=1)
    return fixed_retain_inputs


def edit_model(
    args,
    base_weights,
    pipeline,
    target_concept,
    base_retain_inputs_by_module,
    device="cuda:0",
    max_sequence_length=512,
):
    edit_modules = _select_edit_modules(pipeline)
    module_names = [name for name, _ in edit_modules]
    grouped_modules = {}
    for module_name, module in edit_modules:
        layer_index = int(re.match(r"transformer_blocks\.(\d+)\.", module_name).group(1))
        grouped_modules.setdefault(layer_index, []).append((module_name, module))

    full_token_indices = list(range(max_sequence_length))
    target_token_indices = {target_concept: full_token_indices}
    empty_token_indices = {"": full_token_indices}
    edit_dict = {}

    for _layer_index, layer_modules in sorted(grouped_modules.items()):
        layer_module_names = [module_name for module_name, _module in layer_modules]
        target_traces = _trace_concepts(pipeline, [target_concept], target_token_indices, layer_module_names, args, device, max_sequence_length)
        empty_traces = _trace_concepts(pipeline, [""], empty_token_indices, layer_module_names, args, device, max_sequence_length)
        for module_name, module in layer_modules:
            if module_name not in base_retain_inputs_by_module:
                raise RuntimeError(f"No retain trace for {module_name}")
            target_inputs = target_traces[target_concept][module_name]
            empty_inputs = empty_traces[""][module_name]
            adversarial_inputs = _adversarial_inputs(
                target_inputs,
                base_weights[module_name],
                module.weight.detach(),
                args.adversarial_lambda,
            )
            sum_target_target = adversarial_inputs @ adversarial_inputs.T
            sum_target_anchor = empty_inputs @ adversarial_inputs.T
            weight_before = module.weight.float()
            delta = _closed_form_update(
                sum_target_anchor,
                sum_target_target,
                weight_before,
                args.update_lambda,
                base_retain_inputs_by_module[module_name],
                args.threshold,
            )
            module.weight = torch.nn.Parameter(weight_before.add(delta).to(module.weight.dtype))
            edit_dict[module_name + ".weight"] = module.weight.detach().clone()
            relative_delta = delta.norm() / weight_before.norm().clamp_min(torch.finfo(weight_before.dtype).eps)
            print(f"layer={_layer_index} module={module_name} relative_delta={relative_delta.item():.6e}")
    return edit_dict


def _load_retain_texts(retain_path, heads, target_concept):
    if retain_path is None:
        return [""]
    dataframe = pd.read_csv(retain_path)
    retain_texts = []
    for head in heads.split(","):
        retain_texts.extend(dataframe[head.strip()].dropna().unique().tolist())
    return [
        text for text in retain_texts
        if not re.search(r"\b" + re.escape(target_concept.lower()) + r"\b", text.lower())
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--edited_ckpt", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="logs/checkpoints")
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_concept", type=str, default="nudity")
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default="concept")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--trace_batch_size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--trace_num_steps", type=int, default=4)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=1.0)
    parser.add_argument("--adversarial_lambda", type=float, default=1.0)
    args = parser.parse_args()

    base_pipeline = Flux2KleinPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    base_module_names = [name for name, _module in _select_edit_modules(base_pipeline)]
    base_weights = {
        name: module.weight.detach().cpu().clone()
        for name, module in _select_edit_modules(base_pipeline)
    }
    retain_texts = _load_retain_texts(args.retain_path, args.heads, args.target_concept)
    base_retain_inputs_by_module = _collect_retain_inputs(
        base_pipeline,
        retain_texts,
        base_module_names,
        args,
        args.device,
        512,
    )
    del base_pipeline
    torch.cuda.empty_cache()

    pipeline = Flux2KleinPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    _load_sparse_weights(pipeline, args.edited_ckpt)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    edit_dict = edit_model(
        args,
        base_weights,
        pipeline,
        args.target_concept,
        base_retain_inputs_by_module,
        args.device,
        512,
    )

    os.makedirs(args.save_path, exist_ok=True)
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{args.target_concept}-adversarial"
    save_file(edit_dict, os.path.join(args.save_path, f"{file_name}.safetensors"))
