import os, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import pandas as pd
from safetensors.torch import save_file
from diffusers import Flux2KleinPipeline

FLUX2_MLP_SUFFIX = ".ff_context.linear_out"


def _apply_flux2_chat_template(prompt, tokenizer):
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _subject_token_indices(prompt, tokenizer, max_sequence_length):
    text = _apply_flux2_chat_template(prompt, tokenizer)
    token_inputs = tokenizer(
        text,
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
    if prompt_ids:
        span_length = len(prompt_ids)
        for start_idx in range(0, valid_length - span_length + 1):
            if input_ids[start_idx:start_idx + span_length] == prompt_ids:
                return [start_idx + span_length - 1]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    search_end = input_ids.index(eos_token_id) if eos_token_id in input_ids else valid_length
    content_indices = [idx for idx, token_id in enumerate(input_ids) if int(token_id) not in special_ids and idx < search_end]
    if content_indices:
        return [content_indices[-1]]
    return [valid_length - 1]


def _load_flux_pipeline(model_id, device, torch_dtype):
    if Flux2KleinPipeline is None:
        raise RuntimeError("Flux2KleinPipeline is unavailable in this diffusers install")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def _select_text_mlp_modules(transformer, device):
    selected = []
    layer_pattern = re.compile(r"transformer_blocks\.(\d+)\.")
    for name, module in transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not name.endswith(FLUX2_MLP_SUFFIX):
            continue
        match = layer_pattern.match(name)
        if match is None:
            continue
        selected.append((name, module.to(device)))
    if not selected:
        raise RuntimeError("No Flux2 text MLP modules found: expected transformer_blocks.*.ff_context.linear_out")
    return selected


def _group_mlp_modules_by_layer(edit_modules):
    grouped = {}
    for module_name, module in edit_modules:
        match = re.match(r"transformer_blocks\.(\d+)\.", module_name)
        if match is None:
            continue
        grouped.setdefault(int(match.group(1)), []).append((module_name, module))
    return [(layer_index, grouped[layer_index]) for layer_index in sorted(grouped)]


def _trace_concepts(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):
    module_lookup = dict(pipeline.transformer.named_modules())
    traced_concepts = {}
    trace_batch_size = max(1, int(getattr(args, "trace_batch_size", 1)))
    grouped_concepts = {}

    for concept in dict.fromkeys(concepts):
        token_spec = token_indices.get(concept)
        if not token_spec:
            continue
        selected_token_indices = list(token_spec["indices"])
        pool_selected_tokens = bool(token_spec.get("pool", False))
        if not selected_token_indices:
            continue
        grouped_concepts.setdefault((tuple(selected_token_indices), pool_selected_tokens), []).append(concept)

    for (selected_token_indices, pool_selected_tokens), grouped in grouped_concepts.items():
        selected_token_indices = list(selected_token_indices)
        for start in range(0, len(grouped), trace_batch_size):
            concept_batch = grouped[start:start + trace_batch_size]
            traces = {name: {"inputs": []} for name in module_names}
            handles = []
            for name in module_names:
                module = module_lookup[name]

                def pre_hook(_module, inputs, module_name=name):
                    selected_inputs = inputs[0][:, selected_token_indices, :]
                    if pool_selected_tokens:
                        selected_inputs = selected_inputs.mean(dim=1, keepdim=True)
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
                traced_concepts[concept] = compact
    return traced_concepts


def _concept_matrices(target_inputs, anchor_inputs):
    sum_target_target = [target @ target.T for target in target_inputs]
    sum_target_anchor = [anchor @ target.T for target, anchor in zip(target_inputs, anchor_inputs)]
    return torch.stack(sum_target_anchor).mean(0), torch.stack(sum_target_target).mean(0)


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
    delta = torch.linalg.solve(system.T, residual_projection.T).T
    return delta


def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, device="cuda:0", max_sequence_length=512,):
    edit_modules = _select_text_mlp_modules(
        pipeline.transformer,
        device,
    )
    module_names = [name for name, _ in edit_modules]
    grouped_modules = _group_mlp_modules_by_layer(edit_modules)

    anchor_token_indices = {}
    target_token_indices = {}
    for target_concept, anchor_concept in zip(target_concepts, anchor_concepts):
        target_token_indices[target_concept] = {
            "indices": _subject_token_indices(target_concept, pipeline.tokenizer, max_sequence_length),
            "pool": True,
        }
        anchor_token_indices[anchor_concept] = {
            "indices": _subject_token_indices(anchor_concept, pipeline.tokenizer, max_sequence_length),
            "pool": True,
        }
    retain_token_indices = {}
    for concept in retain_texts:
        if concept == "":
            retain_token_indices[concept] = {
                "indices": list(range(1, max_sequence_length)),
                "pool": False,
            }
        else:
            retain_token_indices[concept] = {
                "indices": _subject_token_indices(concept, pipeline.tokenizer, max_sequence_length),
                "pool": True,
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

    retain_inputs_by_module = {module_name: [] for module_name in module_names}
    for j in range(0, len(retain_texts), args.chunk_size):
        retain_chunk = retain_texts[j:j + args.chunk_size]
        retain_traces = _trace_concepts(pipeline, retain_chunk, retain_token_indices, module_names, args, device, max_sequence_length)
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
        if not retain_inputs_by_module[module_name]:
            raise RuntimeError(f"No retain trace for {module_name}")
        retain_inputs_by_module[module_name] = torch.cat(retain_inputs_by_module[module_name], dim=1)

    edit_dict = {}
    for _, layer_modules in grouped_modules:
        layer_module_names = [module_name for module_name, _module in layer_modules]
        layer_target_traces = _trace_concepts(pipeline, target_concepts, target_token_indices, layer_module_names, args, device, max_sequence_length)
        for module_name, module in layer_modules:
            target_inputs, anchor_inputs = [], []
            for concept, anchor_concept in zip(target_concepts, anchor_concepts):
                concept_trace = layer_target_traces[concept]
                target_inputs.append(concept_trace[module_name]["inputs"])
                anchor_inputs.append(anchor_base_traces[anchor_concept][module_name]["inputs"])

            sum_target_anchor, sum_target_target = _concept_matrices(target_inputs, anchor_inputs)
            sum_target_anchor = sum_target_anchor.to(module.weight.device, torch.float32)
            sum_target_target = sum_target_target.to(module.weight.device, torch.float32)
            retain_inputs = retain_inputs_by_module[module_name]

            weight_before = module.weight.float()
            delta = _closed_form_update(
                sum_target_anchor,
                sum_target_target,
                weight_before,
                args.update_lambda,
                retain_inputs.to(module.weight.device, torch.float32),
                args.threshold,
            )
            module.weight = torch.nn.Parameter(weight_before.add(delta).to(module.weight.dtype))
            edit_dict[module_name + ".weight"] = module.weight.detach().clone()

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
    parser.add_argument("--threshold", type=float, default=3e-2)
    parser.add_argument("--trace_num_steps", type=int, default=20)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=1)
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

    pipeline = _load_flux_pipeline(args.sd_ckpt, args.device, torch.bfloat16)
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
