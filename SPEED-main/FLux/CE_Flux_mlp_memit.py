import os, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
import argparse
import pandas as pd
from safetensors.torch import save_file

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    Flux2KleinPipeline = None


MLP_SUFFIX = ".ff_context.linear_out"


def _apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _subject_token_indices(prompt, tokenizer, max_sequence_length):
    if prompt == "":
        return [0]
    token_inputs = tokenizer(
        _apply_chat_template(prompt, tokenizer),
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    valid_length = int(token_inputs.attention_mask[0].sum().item())
    if valid_length <= 0:
        return [0]
    input_ids = [int(token_id) for token_id in token_inputs.input_ids[0, :valid_length].tolist()]
    try:
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, return_tensors="pt").input_ids[0].tolist()
    except TypeError:
        prompt_ids = tokenizer(prompt, truncation=True, return_tensors="pt").input_ids[0].tolist()
    prompt_ids = [int(token_id) for token_id in prompt_ids]
    for start_idx in range(valid_length - len(prompt_ids) + 1):
        if prompt_ids and input_ids[start_idx:start_idx + len(prompt_ids)] == prompt_ids:
            return list(range(start_idx, start_idx + len(prompt_ids)))
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    search_end = input_ids.index(eos_token_id) if eos_token_id in input_ids else valid_length
    content_indices = [idx for idx, token_id in enumerate(input_ids) if token_id not in special_ids and idx < search_end]
    return content_indices or [valid_length - 1]


def _load_pipeline(model_id, device):
    if Flux2KleinPipeline is None:
        raise RuntimeError("Flux2KleinPipeline is unavailable in this diffusers install")
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    return pipeline


def _select_modules(transformer, device):
    selected = []
    for name, module in transformer.named_modules():
        match = re.match(r"transformer_blocks\.(\d+)\.", name)
        if match is None or not name.endswith(MLP_SUFFIX) or not hasattr(module, "weight") or module.weight is None:
            continue
        selected.append((name, module.to(device)))
    if not selected:
        raise RuntimeError("No Flux2 text MLP modules found: expected transformer_blocks.*.ff_context.linear_out")
    return selected


def _trace_concepts(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):
    module_lookup = dict(pipeline.transformer.named_modules())
    traces_by_concept = {}
    groups = {}
    for concept in dict.fromkeys(concepts):
        token_spec = token_indices.get(concept)
        if token_spec and token_spec["indices"]:
            groups.setdefault((tuple(token_spec["indices"]), bool(token_spec.get("pool", False))), []).append(concept)

    for (indices, pool), concepts_with_indices in groups.items():
        indices = list(indices)
        for start in range(0, len(concepts_with_indices), args.trace_batch_size):
            concept_batch = concepts_with_indices[start:start + args.trace_batch_size]
            traces = {name: {"inputs": [], "outputs": []} for name in module_names}
            handles = []
            for name in module_names:
                module = module_lookup[name]

                def pre_hook(_module, inputs, module_name=name):
                    values = inputs[0][:, indices, :]
                    traces[module_name]["inputs"].append((values.mean(dim=1, keepdim=True) if pool else values).detach().float())

                def output_hook(_module, _inputs, output, module_name=name):
                    values = output[0] if isinstance(output, tuple) else output
                    values = values[:, indices, :]
                    traces[module_name]["outputs"].append((values.mean(dim=1, keepdim=True) if pool else values).detach().float())

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(output_hook))

            generators = [torch.Generator(device=device).manual_seed(args.trace_seed) for _ in concept_batch]
            with torch.no_grad():
                pipeline(
                    prompt=concept_batch,
                    generator=generators,
                    num_inference_steps=args.trace_num_steps,
                    guidance_scale=args.guidance_scale,
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
                        kind: torch.stack(record[kind], dim=0)[:, batch_index, :, :].reshape(-1, torch.stack(record[kind], dim=0).shape[-1]).T
                        for kind in ("inputs", "outputs")
                    }
                    for name, record in traces.items()
                }
    return traces_by_concept


def _concept_matrices(target_inputs, residuals):
    target_target = [target @ target.T for target in target_inputs]
    residual_target = [residual @ target.T for target, residual in zip(target_inputs, residuals)]
    return torch.stack(residual_target).mean(0), torch.stack(target_target).mean(0)


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
    edit_modules = _select_modules(pipeline.transformer, device)
    module_names = [name for name, _ in edit_modules]
    final_module_name = module_names[-1]
    remaining_counts = {name: len(module_names) - index for index, name in enumerate(module_names)}
    target_token_indices = {concept: {"indices": _subject_token_indices(concept, pipeline.tokenizer, max_sequence_length), "pool": True} for concept in target_concepts}
    anchor_token_indices = {concept: {"indices": _subject_token_indices(concept, pipeline.tokenizer, max_sequence_length), "pool": True} for concept in anchor_concepts}
    retain_token_indices = {
        concept: {"indices": list(range(1, max_sequence_length)), "pool": False} if concept == "" else {"indices": _subject_token_indices(concept, pipeline.tokenizer, max_sequence_length), "pool": True}
        for concept in retain_texts
    }

    anchor_final_traces = _trace_concepts(pipeline, anchor_concepts, anchor_token_indices, [final_module_name], args, device, max_sequence_length)
    retain_inputs_by_module = {module_name: [] for module_name in module_names}
    for start in range(0, len(retain_texts), args.chunk_size):
        retain_chunk = retain_texts[start:start + args.chunk_size]
        retain_traces = _trace_concepts(pipeline, retain_chunk, retain_token_indices, module_names, args, device, max_sequence_length)
        for module_name in module_names:
            inputs = [retain_traces[concept][module_name]["inputs"] for concept in retain_chunk if concept in retain_traces]
            if inputs:
                retain_inputs_by_module[module_name].append(torch.cat(inputs, dim=1))
    for module_name in module_names:
        if not retain_inputs_by_module[module_name]:
            raise RuntimeError(f"No retain trace for {module_name}")
        retain_inputs_by_module[module_name] = torch.cat(retain_inputs_by_module[module_name], dim=1)

    edit_dict = {}
    for module_name, module in edit_modules:
        trace_names = [module_name] if module_name == final_module_name else [module_name, final_module_name]
        target_traces = _trace_concepts(pipeline, target_concepts, target_token_indices, trace_names, args, device, max_sequence_length)
        target_inputs, residuals = [], []
        for concept, anchor_concept in zip(target_concepts, anchor_concepts):
            current_final = target_traces[concept][final_module_name]["outputs"]
            anchor_final = anchor_final_traces[anchor_concept][final_module_name]["outputs"].to(current_final.device, current_final.dtype)
            target_inputs.append(target_traces[concept][module_name]["inputs"])
            residuals.append((anchor_final - current_final) / remaining_counts[module_name])

        residual_target, target_target = _concept_matrices(target_inputs, residuals)
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
    parser.add_argument("--sd_ckpt", type=str, default="black-forest-labs/FLUX.2-klein-4B")
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
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--update_lambda", type=float, default=1.0)
    args = parser.parse_args()

    target_concepts = [concept.strip() for concept in args.target_concepts.split(",")]
    anchor_concepts = [concept.strip() for concept in args.anchor_concepts.split(",")]
    if not target_concepts or any(not concept for concept in target_concepts):
        raise ValueError("--target_concepts must not contain empty concepts")
    if len(anchor_concepts) == 1:
        anchor_concepts *= len(target_concepts)
    if len(anchor_concepts) != len(target_concepts):
        raise ValueError("--anchor_concepts must contain one anchor or one per target")
    retain_texts = [""]
    if args.retain_path:
        if args.heads is None:
            raise ValueError("--heads is required when --retain_path is provided")
        retain_texts = []
        frame = pd.read_csv(args.retain_path)
        for head in args.heads.split(","):
            retain_texts.extend(frame[head.strip()].dropna().unique().tolist())
    excluded = target_concepts + [concept for concept in anchor_concepts if concept]
    retain_texts = [text for text in retain_texts if not any(re.search(r"\b" + re.escape(concept.lower()) + r"\b", str(text).lower()) for concept in excluded)]

    pipeline = _load_pipeline(args.sd_ckpt, args.device)
    edit_dict = edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, args.device)
    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{'_'.join(target_concepts[:5])}_{len(target_concepts)}-mlp-memit"
    os.makedirs(save_path, exist_ok=True)
    save_file(edit_dict, os.path.join(save_path, f"{file_name}.safetensors"))
