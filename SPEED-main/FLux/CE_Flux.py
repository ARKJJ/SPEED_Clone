import os, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import pandas as pd
from diffusers import DiffusionPipeline
from safetensors.torch import save_file


ATTENTION_SUFFIXES = {
    "Q": ".attn.add_q_proj",
    "K": ".attn.add_k_proj",
    "V": ".attn.add_v_proj",
}


def get_token_id(prompt, tokenizer=None, max_sequence_length=None, return_ids_only=True):
    token_ids = tokenizer(prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids


def _select_text_attention_modules(transformer, device, params):
    selected = []
    selected_suffixes = [ATTENTION_SUFFIXES[param] for param in params]
    for name, module in transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        module_suffix = next((suffix for suffix in selected_suffixes if suffix in name), None)
        if module_suffix is None:
            continue
        if name.startswith("transformer_blocks."):
            selected.append((name, module.to(device), module_suffix))
    return selected, selected_suffixes


def _trace_concepts(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):
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
                    traces[module_name]["inputs"].append(inputs[0][:, selected_token_indices, :].detach().float())

                def out_hook(_module, _inputs, output, module_name=name):
                    output = output[0] if isinstance(output, tuple) else output
                    traces[module_name]["outputs"].append(output[:, selected_token_indices, :].detach().float())

                handles.extend([module.register_forward_pre_hook(pre_hook), module.register_forward_hook(out_hook)])

            generators = [
                torch.Generator(device=device).manual_seed(args.trace_seed)
                for _ in concept_batch
            ]
            with torch.no_grad():
                pipeline(
                    concept_batch,
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
    return traced_concepts


def _closed_form_update(keys, residuals, update_lambda, retain_inputs, retain_threshold=1e-1):
    retain_inputs = retain_inputs.to(device=keys.device, dtype=keys.dtype)
    covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
    null_basis = U[:, S < retain_threshold]
    if null_basis.shape[1] == 0:
        projector = torch.eye(keys.shape[0], device=keys.device, dtype=keys.dtype)
    else:
        projector = null_basis @ null_basis.T
    projected_keys = projector @ keys
    eye = torch.eye(keys.shape[0], device=keys.device, dtype=keys.dtype)
    system = keys @ projected_keys.T + update_lambda * eye
    delta = torch.linalg.solve(system.T, (residuals @ projected_keys.T).T).T
    return delta 


def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, device="cuda:0", max_sequence_length=512,):
    edit_modules, selected_suffixes = _select_text_attention_modules(pipeline.transformer, device, args.params)
    module_names = [name for name, _, _ in edit_modules]
    edit_module_suffixes = [suffix for _, _, suffix in edit_modules]
    final_modules = {}
    remaining_counts = [0] * len(module_names)
    suffix_counts = {}
    for module_index in range(len(module_names) - 1, -1, -1):
        suffix = edit_module_suffixes[module_index]
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        remaining_counts[module_index] = suffix_counts[suffix]
        final_modules.setdefault(suffix, module_names[module_index])

    anchor_token_indices = {}
    target_token_indices = {}
    for target_concept, anchor_concept in zip(target_concepts, anchor_concepts):
        target_inputs = get_token_id(target_concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)
        anchor_inputs = get_token_id(anchor_concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)
        if target_concepts == ["nudity"]:
            target_token_indices[target_concept] = list(range(0, target_inputs.input_ids.shape[1]))
            anchor_token_indices[anchor_concept] = list(range(0, anchor_inputs.input_ids.shape[1]))
        else:
            target_token_indices[target_concept] = [int(target_inputs.attention_mask[0].argmax().item())]
            anchor_token_indices[anchor_concept] = [int(anchor_inputs.attention_mask[0].argmax().item())]

    retain_token_indices = {}
    for concept in retain_texts:
        concept_inputs = get_token_id(concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)
        if concept == "":
            retain_token_indices[concept] = list(range(0, concept_inputs.input_ids.shape[1]))
        else:
            retain_token_indices[concept] = [int(concept_inputs.attention_mask[0].argmax().item())]

    print("\nSelected text-side attention modules:")
    for name in module_names:
        print(f"  {name}")

    final_module_names = [final_modules[suffix] for suffix in selected_suffixes if suffix in final_modules]
    anchor_final_traces = _trace_concepts(pipeline, anchor_concepts, anchor_token_indices, final_module_names, args, device, max_sequence_length)

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
    for module_index, (module_name, module, suffix) in enumerate(edit_modules):
        final_module_name = final_modules[suffix]

        trace_module_names = list(dict.fromkeys([module_name, final_module_name]))
        edit_traces = _trace_concepts(pipeline, target_concepts, target_token_indices, trace_module_names, args, device, max_sequence_length)
        remaining_count = remaining_counts[module_index]
        keys, residuals = [], []
        for concept, anchor_concept in zip(target_concepts, anchor_concepts):
            concept_trace = edit_traces[concept]
            final_current = concept_trace[final_module_name]["outputs"]
            anchor = anchor_final_traces[anchor_concept][final_module_name]["outputs"].to(final_current.device, final_current.dtype)
            keys.append(concept_trace[module_name]["inputs"])
            residuals.append((anchor - final_current) * (args.residual_scale / remaining_count))

        keys = torch.cat(keys, dim=1).to(module.weight.device, torch.float32)
        residuals = torch.cat(residuals, dim=1).to(module.weight.device, torch.float32)
        retain_inputs = retain_inputs_by_module[module_name]

        delta = _closed_form_update(keys, residuals, args.update_lambda * len(target_concepts) * args.trace_num_steps, retain_inputs.to(module.weight.device, torch.float32), args.threshold)
        module.weight = torch.nn.Parameter(module.weight.float().add(delta).to(module.weight.dtype))
        edit_dict[module_name + ".weight"] = module.weight.detach().clone()
        print(f"  Updated {module_name} | ||delta||={delta.norm().item():.4f}")

    print(f"Current model status: Edited {target_concepts} into {anchor_concepts or ['null-anchor']}")
    return edit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", help="base version for FLUX", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--trace_batch_size", type=int, default=4)
    parser.add_argument("--params", type=str, default="KV", choices=["Q", "K", "V", "QK", "KV", "QKV"])
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--trace_num_steps", type=int, default=20)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    args = parser.parse_args()

    target_concepts = [con.strip() for con in args.target_concepts.split(",")]
    if not target_concepts or any(concept == "" for concept in target_concepts):
        raise ValueError("--target_concepts must not contain empty concepts")
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path

    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}"
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
