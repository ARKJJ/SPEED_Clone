import argparse
import gc
import os
import time
from dataclasses import dataclass

import torch

torch.set_grad_enabled(False)

from diffusers import DiffusionPipeline
from safetensors.torch import save_file


@dataclass
class MemitFluxConfig:
    layer_start: int = 6
    layer_end: int = 15
    layer_stride: int = 2
    trace_num_steps: int = 4
    trace_seed: int = 0
    trace_resolution: int = 512
    null_anchor_mode: str = "preserve_mean"
    preserve_lambda: float = 1.0
    update_lambda: float = 1e-4
    residual_scale: float = 1.0


def get_token_id(prompt, tokenizer=None, max_sequence_length=None, return_ids_only=True):
    token_ids = tokenizer(prompt,
                          padding="max_length",
                          max_length=max_sequence_length or tokenizer.model_max_length,
                          truncation=True,
                          return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids


def _parse_concepts(raw_value):
    return [] if not raw_value else [value.strip() for value in raw_value.split(";") if value.strip()]


def _parse_replace_indices(edit_concepts, replace_indices_arg):
    if replace_indices_arg is None:
        return [None] * len(edit_concepts)
    replace_indices = [
        None if value.strip() == "" or value.strip().lower() == "all"
        else [int(index.strip()) for index in value.split(",") if index.strip()]
        for value in replace_indices_arg.split(";")
    ]
    if len(replace_indices) != len(edit_concepts):
        raise ValueError("replace_indices length must match edit_concepts length")
    return replace_indices


def _expand_concepts(edit_concepts, guide_concepts, concept_type, expand_prompts):
    edits, guides = list(edit_concepts), list(guide_concepts)
    if expand_prompts != "true":
        return edits, guides
    templates = (
        ["painting by {}", "art by {}", "artwork by {}", "picture by {}", "style of {}"]
        if concept_type == "art"
        else ["image of {}", "photo of {}", "portrait of {}", "picture of {}", "painting of {}"]
    )
    for edit_concept, guide_concept in zip(edit_concepts, guide_concepts):
        edits.extend(template.format(edit_concept) for template in templates)
        guides.extend(template.format(guide_concept) for template in templates)
    return edits, guides


def _layer_indices(layer_start, layer_end, layer_stride):
    if layer_stride <= 0:
        raise ValueError("layer_stride must be positive")
    if layer_end <= layer_start:
        raise ValueError("layer_end must be greater than layer_start")
    return list(range(layer_start, layer_end, layer_stride))


def _select_text_qk_modules(transformer, device, layer_indices):
    selected, allowed_layers = [], set(layer_indices)
    for name, module in transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if ".attn.add_q_proj" not in name and ".attn.add_k_proj" not in name:
            continue
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "transformer_blocks" and int(parts[1]) in allowed_layers:
            selected.append((name, module.to(device)))
    return selected


def _content_token_indices(pipeline, concept, requested_indices, max_sequence_length):
    tokens = get_token_id(concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)
    token_count = max(int(tokens.attention_mask.sum().item()) - 1, 0)
    if requested_indices is None:
        return list(range(token_count))
    return [index for index in requested_indices if 0 <= index < token_count]


def _trace_prompt(pipeline, prompt, token_indices, module_names, config, device, max_sequence_length):
    module_lookup = dict(pipeline.transformer.named_modules())
    traces = {name: {"inputs": [], "outputs": []} for name in module_names}
    handles = []

    for name in module_names:
        module = module_lookup[name]

        def pre_hook(_module, inputs, module_name=name):
            traces[module_name]["inputs"].append(inputs[0][:, token_indices, :].detach().float())

        def out_hook(_module, _inputs, output, module_name=name):
            output = output[0] if isinstance(output, tuple) else output
            traces[module_name]["outputs"].append(output[:, token_indices, :].detach().float())

        handles.extend([module.register_forward_pre_hook(pre_hook), module.register_forward_hook(out_hook)])

    try:
        generator = torch.Generator(device=device).manual_seed(config.trace_seed)
        with torch.no_grad():
            pipeline(
                prompt,
                generator=generator,
                num_inference_steps=config.trace_num_steps,
                guidance_scale=0.0,
                height=config.trace_resolution,
                width=config.trace_resolution,
                max_sequence_length=max_sequence_length,
                output_type="latent",
            )
    finally:
        for handle in handles:
            handle.remove()

    compact = {}
    for name, record in traces.items():
        if not record["inputs"] or not record["outputs"]:
            raise RuntimeError(f"No trace was collected for module '{name}'")
        compact[name] = {
            "inputs": torch.cat(record["inputs"], dim=0).reshape(-1, record["inputs"][0].shape[-1]).T,
            "outputs": torch.cat(record["outputs"], dim=0).reshape(-1, record["outputs"][0].shape[-1]).T,
        }
    return compact


def _closed_form_update(keys, residuals, preserve_gram, update_lambda, preserve_lambda):
    eye = torch.eye(keys.shape[0], device=keys.device, dtype=keys.dtype)
    system = keys @ keys.T + update_lambda * eye
    if preserve_gram is not None:
        system = system + preserve_lambda * preserve_gram.to(device=keys.device, dtype=keys.dtype)
    return torch.linalg.solve(system.T, (residuals @ keys.T).T).T


def _trace_many(pipeline, concepts, token_indices, module_names, config, device, max_sequence_length):
    return {
        concept: _trace_prompt(pipeline, concept, token_indices[concept], module_names, config, device, max_sequence_length)
        for concept in concepts
        if token_indices.get(concept)
    }


def _mean_outputs(traces, concepts, module_name):
    outputs = [traces[c][module_name]["outputs"] for c in concepts if c in traces and module_name in traces[c]]
    return None if not outputs else torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)


def edit_model(
    config,
    pipeline,
    target_concepts,
    anchor_concepts,
    retain_texts,
    replace_indices=None,
    device="cuda:0",
    max_sequence_length=256,
):
    layer_ids = _layer_indices(config.layer_start, config.layer_end, config.layer_stride)
    edit_modules = _select_text_qk_modules(pipeline.transformer, device, layer_ids)
    if not edit_modules:
        raise RuntimeError("No text-side q/k modules were selected for editing")
    if replace_indices is None:
        replace_indices = [None] * len(target_concepts)
    if len(replace_indices) != len(target_concepts):
        raise ValueError("replace_indices length must match target_concepts length")

    module_names = [name for name, _ in edit_modules]
    anchor_concepts = anchor_concepts if anchor_concepts else retain_texts
    if not anchor_concepts:
        raise ValueError("anchor_concepts is empty and null-anchor requires retain_texts")

    # region [Target and Anchor]
    token_indices = {
        concept: _content_token_indices(pipeline, concept, None, max_sequence_length)
        for concept in anchor_concepts + retain_texts
    }
    for concept, indices in zip(target_concepts, replace_indices):
        token_indices[concept] = _content_token_indices(pipeline, concept, indices, max_sequence_length)

    print("\nSelected text-side q/k modules:")
    for name in module_names:
        print(f"  {name}")

    anchor_traces = _trace_many(pipeline, anchor_concepts, token_indices, module_names, config, device, max_sequence_length)
    anchor_target_means = {
        module_name: _mean_outputs(anchor_traces, anchor_concepts, module_name)
        for module_name in module_names
    }
    # endregion

    # region [Retain]
    retain_traces = _trace_many(pipeline, retain_texts, token_indices, module_names, config, device, max_sequence_length)
    preserve_grams = {}
    for module_name in module_names:
        retain_inputs = [
            retain_traces[concept][module_name]["inputs"]
            for concept in retain_texts
            if concept in retain_traces and module_name in retain_traces[concept]
        ]
        preserve_grams[module_name] = None if not retain_inputs else (torch.cat(retain_inputs, dim=1) @ torch.cat(retain_inputs, dim=1).T)
    # endregion

    edit_dict = {}

    # region [Layer Update]
    for module_index, (module_name, module) in enumerate(edit_modules):
        target_mean = anchor_target_means[module_name]
        if target_mean is None:
            print(f"  Warning: no anchor trace for {module_name}, skipping.")
            continue

        edit_traces = _trace_many(pipeline, target_concepts, token_indices, module_names, config, device, max_sequence_length)
        keys, residuals = [], []
        for concept in target_concepts:
            if concept not in edit_traces or module_name not in edit_traces[concept]:
                continue
            current = edit_traces[concept][module_name]["outputs"]
            target = target_mean.to(current.device, current.dtype).expand(-1, current.shape[1])
            keys.append(edit_traces[concept][module_name]["inputs"])
            residuals.append((target - current) * (config.residual_scale / max(len(edit_modules) - module_index, 1)))
        if not keys:
            print(f"  Warning: no edit trace for {module_name}, skipping.")
            continue

        preserve_gram = preserve_grams[module_name]
        keys = torch.cat(keys, dim=1).to(module.weight.device, torch.float32)
        residuals = torch.cat(residuals, dim=1).to(module.weight.device, torch.float32)
        if preserve_gram is not None:
            preserve_gram = preserve_gram.to(module.weight.device, torch.float32)

        delta = _closed_form_update(keys, residuals, preserve_gram, config.update_lambda, config.preserve_lambda)
        module.weight = torch.nn.Parameter(module.weight.float().add(delta).to(module.weight.dtype))
        edit_dict[module_name + ".weight"] = module.weight.detach().clone()
        print(f"  Updated {module_name} | ||delta||={delta.norm().item():.4f}")
    # endregion

    print(f"Current model status: Edited {target_concepts} into {anchor_concepts or ['null-anchor']}")
    return edit_dict


def apply_memit_flux(
    model_id,
    edit_concepts,
    guide_concepts,
    preserve_concepts,
    save_dir,
    exp_name,
    torch_dtype,
    device,
    max_sequence_length,
    replace_indices=None,
    config=None,
):
    config = config or MemitFluxConfig()
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None).to(device)
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
    start_time = time.time()
    edit_dict = edit_model(config, pipeline, edit_concepts, guide_concepts, preserve_concepts, replace_indices, device, max_sequence_length)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, exp_name + ".safetensors")
    save_file(edit_dict, save_path)
    pipeline = None
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    gc.collect()
    print(f"\n\nErased concepts using MEMIT-style FLUX q/k editing\nModel edited in {time.time() - start_time:.2f} seconds\nWeights saved to {save_path}\n")
    return save_path


UCE_double_proxy = apply_memit_flux


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MEMITFlux", description="MEMIT-style concept erasure for FLUX text-side q/k")
    parser.add_argument("--edit_concepts", type=str, default=None)
    parser.add_argument("--target_concepts", type=str, default=None)
    parser.add_argument("--guide_concepts", type=str, default=None)
    parser.add_argument("--anchor_concepts", type=str, default=None)
    parser.add_argument("--preserve_concepts", type=str, default=None)
    parser.add_argument("--retain_concepts", type=str, default=None)
    parser.add_argument("--concept_type", type=str, required=True, choices=["art", "object"])
    parser.add_argument("--replace_indices", type=str, default="all")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--expand_prompts", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--layer_start", type=int, default=6)
    parser.add_argument("--layer_end", type=int, default=15)
    parser.add_argument("--layer_stride", type=int, default=2)
    parser.add_argument("--trace_num_steps", type=int, default=4)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--null_anchor_mode", type=str, default="preserve_mean", choices=["preserve_mean"])
    parser.add_argument("--preserve_lambda", type=float, default=1.0)
    parser.add_argument("--update_lambda", type=float, default=1e-4)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    args = parser.parse_args()

    target_arg = args.edit_concepts or args.target_concepts
    if not target_arg:
        parser.error("--edit_concepts or --target_concepts is required")
    anchor_arg = args.guide_concepts if args.guide_concepts is not None else args.anchor_concepts
    retain_arg = args.preserve_concepts if args.preserve_concepts is not None else args.retain_concepts

    edit_concepts = _parse_concepts(target_arg)
    guide_concepts = _parse_concepts(anchor_arg or ("art" if args.concept_type == "art" else ""))
    preserve_concepts = _parse_concepts(retain_arg)
    replace_indices = _parse_replace_indices(edit_concepts, args.replace_indices)
    edit_concepts, guide_concepts = _expand_concepts(edit_concepts, guide_concepts, args.concept_type, args.expand_prompts)
    replace_indices += [None] * max(len(edit_concepts) - len(replace_indices), 0)
    config = MemitFluxConfig(
        args.layer_start,
        args.layer_end,
        args.layer_stride,
        args.trace_num_steps,
        args.trace_seed,
        args.trace_resolution,
        args.null_anchor_mode,
        args.preserve_lambda,
        args.update_lambda,
        args.residual_scale,
    )

    print(f"\nErasing  : {edit_concepts}")
    print(f"Guiding  : {guide_concepts}")
    print(f"Preserving: {preserve_concepts}\n")
    apply_memit_flux(
        model_id=args.model_id,
        edit_concepts=edit_concepts,
        guide_concepts=guide_concepts,
        preserve_concepts=preserve_concepts,
        save_dir=args.save_dir,
        exp_name=args.exp_name or "flux_memit_qk_test",
        torch_dtype=torch.bfloat16,
        device=args.device,
        max_sequence_length=256 if "schnell" in args.model_id else 512,
        replace_indices=replace_indices,
        config=config,
    )
