import os, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import pandas as pd
from diffusers import DiffusionPipeline
from safetensors.torch import save_file

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    Flux2KleinPipeline = None


OLD_MLP2_SUFFIX = ".ff_context.net.2"
FLUX2_MLP2_SUFFIX = ".ff_context.linear_out"
COMMUNITY_FLUX2_MLP2_SUFFIX = ".txt_mlp.2"


def get_token_id(prompt, tokenizer=None, max_sequence_length=None, return_ids_only=True):
    token_ids = tokenizer(prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids


def _find_last_token_subsequence(sequence, subsequence):
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start + len(subsequence) - 1
    return None


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
    if prompt == "":
        return [0]

    text = _apply_flux2_chat_template(prompt, tokenizer)
    token_inputs = get_token_id(text, tokenizer, max_sequence_length, return_ids_only=False)
    valid_length = int(token_inputs.attention_mask[0].sum().item())
    if valid_length <= 0:
        return [0]

    input_ids = [int(token_id) for token_id in token_inputs.input_ids[0, :valid_length].tolist()]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    content_inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    content_ids = [int(token_id) for token_id in content_inputs.input_ids[0].tolist() if int(token_id) not in special_ids]
    selected = _find_last_token_subsequence(input_ids, content_ids)
    if selected is not None:
        start = selected - len(content_ids) + 1
        return list(range(start, selected + 1))

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    search_end = input_ids.index(eos_token_id) if eos_token_id in input_ids else valid_length
    content_indices = [idx for idx, token_id in enumerate(input_ids) if int(token_id) not in special_ids and idx < search_end]
    if content_indices:
        return [content_indices[-1]]
    return [valid_length - 1]


def _normalize_token_spec(token_spec):
    if isinstance(token_spec, dict):
        return list(token_spec["indices"]), bool(token_spec.get("pool", False))
    return list(token_spec), False


def _load_flux_pipeline(model_id, device, torch_dtype):
    model_id_lower = model_id.lower()
    if "flux.2-klein" in model_id_lower:
        if Flux2KleinPipeline is None:
            raise RuntimeError(
                "Flux2KleinPipeline is unavailable in this diffusers install. "
                "Upgrade diffusers to a version that includes FLUX.2 support."
            )
        pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch_dtype).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def _resolve_mlp_spec(transformer):
    module_names = list(dict(transformer.named_modules()).keys())
    if any(name.endswith(FLUX2_MLP2_SUFFIX) for name in module_names):
        return {
            "suffix": FLUX2_MLP2_SUFFIX,
            "layer_pattern": r"transformer_blocks\.(\d+)\.",
            "label": "FLUX.2 text MLP output",
        }
    if any(name.endswith(OLD_MLP2_SUFFIX) for name in module_names):
        return {
            "suffix": OLD_MLP2_SUFFIX,
            "layer_pattern": r"transformer_blocks\.(\d+)\.",
            "label": "FLUX text-side MLP2",
        }
    if any(name.endswith(COMMUNITY_FLUX2_MLP2_SUFFIX) for name in module_names):
        return {
            "suffix": COMMUNITY_FLUX2_MLP2_SUFFIX,
            "layer_pattern": r"double_blocks\.(\d+)\.",
            "label": "community FLUX.2 text MLP",
        }
    raise RuntimeError(
        "No recognized FLUX text MLP modules found. "
        "Expected one of 'transformer_blocks.*.ff_context.linear_out', "
        "'transformer_blocks.*.ff_context.net.2', or 'double_blocks.*.txt_mlp.2'."
    )


def _select_text_mlp_modules(transformer, device, args):
    spec = _resolve_mlp_spec(transformer)
    selected = []
    layer_start = int(getattr(args, "layer_start", 0))
    layer_end = getattr(args, "layer_end", None)
    layer_stride = max(1, int(getattr(args, "layer_stride", 1)))
    if layer_end is not None:
        layer_end = int(layer_end)
    layer_pattern = re.compile(spec["layer_pattern"])
    for name, module in transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not name.endswith(spec["suffix"]):
            continue
        match = layer_pattern.match(name)
        if match is None:
            continue
        layer_index = int(match.group(1))
        if layer_index < layer_start:
            continue
        if layer_end is not None and layer_index > layer_end:
            continue
        if (layer_index - layer_start) % layer_stride != 0:
            continue
        selected.append((name, module.to(device)))
    return selected, spec


def _group_mlp_modules_by_layer(edit_modules):
    grouped = {}
    for module_name, module in edit_modules:
        match = re.match(r"(?:transformer_blocks|double_blocks)\.(\d+)\.", module_name)
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
        selected_token_indices, pool_selected_tokens = _normalize_token_spec(token_spec)
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
                    compact[name] = {
                        "inputs": input_steps[:, batch_index, :, :].reshape(-1, input_steps.shape[-1]).T,
                    }
                traced_concepts[concept] = compact
    return traced_concepts


def _closed_form_update(target_inputs, anchor_inputs, weight, update_lambda, retain_inputs, retain_threshold=1e-1):
    retain_inputs = retain_inputs.to(device=target_inputs.device, dtype=target_inputs.dtype)
    covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
    null_basis = U[:, S < retain_threshold]
    if null_basis.shape[1] == 0:
        projector = torch.eye(target_inputs.shape[0], device=target_inputs.device, dtype=target_inputs.dtype)
    else:
        projector = null_basis @ null_basis.T
    eye = torch.eye(target_inputs.shape[0], device=target_inputs.device, dtype=target_inputs.dtype)
    delta = weight @ (anchor_inputs - target_inputs) @ target_inputs.T @ projector @ (target_inputs @ target_inputs.T @ projector + update_lambda * eye).inverse()
    return delta


def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, device="cuda:0", max_sequence_length=512,):
    edit_modules, mlp_spec = _select_text_mlp_modules(
        pipeline.transformer,
        device,
        args,
    )
    if not edit_modules:
        raise RuntimeError(f"No text-side MLP modules selected for {mlp_spec['label']}")
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
    print(f"\nSelected {mlp_spec['label']} modules:")
    for name in module_names:
        print(f"  {name}")

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
    for layer_index, layer_modules in grouped_modules:
        layer_module_names = [module_name for module_name, _module in layer_modules]
        layer_target_traces = _trace_concepts(pipeline, target_concepts, target_token_indices, layer_module_names, args, device, max_sequence_length)
        layer_anchor_traces = _trace_concepts(pipeline, anchor_concepts, anchor_token_indices, layer_module_names, args, device, max_sequence_length)
        for module_name, module in layer_modules:
            target_inputs, anchor_inputs = [], []
            for concept, anchor_concept in zip(target_concepts, anchor_concepts):
                concept_trace = layer_target_traces[concept]
                target_inputs.append(concept_trace[module_name]["inputs"])
                anchor_inputs.append(layer_anchor_traces[anchor_concept][module_name]["inputs"])

            target_inputs = torch.cat(target_inputs, dim=1).to(module.weight.device, torch.float32)
            anchor_inputs = torch.cat(anchor_inputs, dim=1).to(module.weight.device, torch.float32)
            retain_inputs = retain_inputs_by_module[module_name]

            lambda_eff = args.update_lambda * target_inputs.shape[1]
            weight_before = module.weight.float()
            delta = _closed_form_update(
                target_inputs,
                anchor_inputs,
                weight_before,
                lambda_eff,
                retain_inputs.to(module.weight.device, torch.float32),
                args.threshold,
            )
            weight_norm = weight_before.norm()
            delta_norm = delta.norm()
            input_diff_norm = (target_inputs - anchor_inputs).norm()
            anchor_projected = weight_before @ anchor_inputs
            module.weight = torch.nn.Parameter(weight_before.add(delta).to(module.weight.dtype))
            target_projected_after = module.weight.float() @ target_inputs
            fit_gap_norm = (target_projected_after - anchor_projected).norm()
            edit_dict[module_name + ".weight"] = module.weight.detach().clone()
            print(
                f"  Updated layer={layer_index} {module_name} | "
                f"samples={target_inputs.shape[1]} | "
                f"retain_samples={retain_inputs.shape[1]} | "
                f"lambda_eff={lambda_eff:.6f} | "
                f"||delta||={delta_norm.item():.4f} | "
                f"||W||={weight_norm.item():.4f} | "
                f"rel={(delta_norm / (weight_norm + 1e-12)).item():.6f} | "
                f"input_diff_norm={input_diff_norm.item():.4f} | "
                f"fit_gap_norm={fit_gap_norm.item():.4f} | "
                f"fit_rel={(fit_gap_norm / (anchor_projected.norm() + 1e-12)).item():.6f}"
            )

    print(f"Current model status: Edited {target_concepts} into {anchor_concepts or ['null-anchor']}")
    return edit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", help="base version for FLUX", type=str, default="black-forest-labs/FLUX.2-klein-4B")
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
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--layer_stride", type=int, default=1)
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
