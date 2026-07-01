import os, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import argparse
import pandas as pd
from diffusers import DiffusionPipeline
from safetensors.torch import save_file


def get_token_id(prompt, tokenizer=None, max_sequence_length=None, return_ids_only=True):
    token_ids = tokenizer(prompt,padding="max_length",max_length=max_sequence_length or tokenizer.model_max_length,truncation=True,return_tensors="pt")
    return token_ids.input_ids if return_ids_only else token_ids


def _attention_suffixes(params):
    param_map = {
        "Q": ".attn.add_q_proj",
        "K": ".attn.add_k_proj",
        "V": ".attn.add_v_proj",
    }
    params = params.upper()
    return [param_map[param] for param in "QKV" if param in params]


def _attention_suffix(module_name):
    for suffix in _attention_suffixes("QKV"):
        if suffix in module_name:
            return suffix
    raise ValueError(f"Unsupported attention module name: {module_name}")


def _final_modules_by_suffix(module_names):
    final_modules = {}
    for module_name in module_names:
        final_modules[_attention_suffix(module_name)] = module_name
    return final_modules


def _remaining_modules_with_suffix(module_names, module_index, suffix):
    return sum(1 for name in module_names[module_index:] if _attention_suffix(name) == suffix)


def _select_text_attention_modules(transformer, device, params):#选择所有transformer block里的text-side attention模块
    selected = []
    suffixes = _attention_suffixes(params)
    for name, module in transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not any(suffix in name for suffix in suffixes):
            continue
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "transformer_blocks":
            selected.append((name, module.to(device)))
    return selected


def _trace_prompt(pipeline, prompt, token_indices, module_names, args, device, max_sequence_length):
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

    try:#此处待删除
        generator = torch.Generator(device=device).manual_seed(args.trace_seed)
        with torch.no_grad():
            pipeline(
                prompt,
                generator=generator,
                num_inference_steps=args.trace_num_steps,
                guidance_scale=0.0,
                height=args.trace_resolution,
                width=args.trace_resolution,
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
    eye = torch.eye(projected_keys.shape[0], device=projected_keys.device, dtype=projected_keys.dtype)
    system = projected_keys @ projected_keys.T + update_lambda * eye#？
    delta = torch.linalg.solve(system.T, (residuals @ projected_keys.T).T).T
    return delta @ projector


def _trace_many(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):
    return {
        concept: _trace_prompt(pipeline, concept, token_indices[concept], module_names, args, device, max_sequence_length)
        for concept in dict.fromkeys(concepts)
        if token_indices.get(concept)
    }


def _mean_outputs(traces, concepts, module_name):
    outputs = [traces[c][module_name]["outputs"] for c in concepts if c in traces and module_name in traces[c]]
    return None if not outputs else torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)


def edit_model(args,pipeline,target_concepts,anchor_concepts,retain_texts,device="cuda:0",max_sequence_length=256,):
    edit_modules = _select_text_attention_modules(pipeline.transformer, device, args.params)
    module_names = [name for name, _ in edit_modules]

    retain_texts = [
        text for text in retain_texts
        if not any(re.search(r"\b" + re.escape(concept.lower()) + r"\b", text.lower()) for concept in target_concepts)
    ]
    if len(retain_texts) + len(target_concepts) != len(set(retain_texts + target_concepts)):
        raise ValueError("retain_texts and target_concepts must not overlap")

    # region [Target and Anchor]
    token_indices = {}
    for concept in anchor_concepts + retain_texts:
        concept_inputs = get_token_id(concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)
        token_count = max(int(concept_inputs.attention_mask.sum().item()) - 1, 0)
        token_indices[concept] = [0] if concept == "" and token_count == 0 else list(range(token_count))
    for concept in target_concepts:
        concept_inputs = get_token_id(concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)
        token_count = max(int(concept_inputs.attention_mask.sum().item()) - 1, 0)
        token_indices[concept] = list(range(token_count))

    print("\nSelected text-side attention modules:")
    for name in module_names:
        print(f"  {name}")

    final_modules = _final_modules_by_suffix(module_names)
    final_module_names = list(dict.fromkeys(final_modules.values()))
    anchor_final_traces = _trace_many(pipeline, anchor_concepts, token_indices, final_module_names, args, device, max_sequence_length)
    anchor_final_means = {
        module_name: _mean_outputs(anchor_final_traces, anchor_concepts, module_name)
        for module_name in final_module_names
    }
    # endregion

    # region [Retain]
    retain_traces = _trace_many(pipeline, retain_texts, token_indices, module_names, args, device, max_sequence_length)
    retain_inputs_by_module = {}
    for module_name in module_names:
        retain_inputs = [
            retain_traces[concept][module_name]["inputs"]
            for concept in retain_texts
            if concept in retain_traces and module_name in retain_traces[concept]
        ]
        if not retain_inputs:
            raise RuntimeError(f"No retain trace for {module_name}")
        retain_inputs_by_module[module_name] = torch.cat(retain_inputs, dim=1)
    # endregion

    edit_dict = {}

    # region [Layer Update]
    for module_index, (module_name, module) in enumerate(edit_modules):
        suffix = _attention_suffix(module_name)
        final_module_name = final_modules[suffix]
        anchor_final_mean = anchor_final_means[final_module_name]
        if anchor_final_mean is None:
            print(f"  Warning: no final anchor trace for {final_module_name}, skipping {module_name}.")
            continue

        trace_module_names = list(dict.fromkeys([module_name, final_module_name]))
        edit_traces = _trace_many(pipeline, target_concepts, token_indices, trace_module_names, args, device, max_sequence_length)
        remaining_count = max(_remaining_modules_with_suffix(module_names, module_index, suffix), 1)
        keys, residuals = [], []
        for concept in target_concepts:
            if concept not in edit_traces or module_name not in edit_traces[concept] or final_module_name not in edit_traces[concept]:
                continue
            final_current = edit_traces[concept][final_module_name]["outputs"]
            target = anchor_final_mean.to(final_current.device, final_current.dtype).expand(-1, final_current.shape[1])
            keys.append(edit_traces[concept][module_name]["inputs"])
            residuals.append((target - final_current) * (args.residual_scale / remaining_count))
        if not keys:
            print(f"  Warning: no edit trace for {module_name}, skipping.")
            continue

        keys = torch.cat(keys, dim=1).to(module.weight.device, torch.float32)
        residuals = torch.cat(residuals, dim=1).to(module.weight.device, torch.float32)
        retain_inputs = retain_inputs_by_module[module_name]

        delta = _closed_form_update(keys,residuals,args.update_lambda,retain_inputs.to(module.weight.device, torch.float32),args.threshold,)
        module.weight = torch.nn.Parameter(module.weight.float().add(delta).to(module.weight.dtype))
        edit_dict[module_name + ".weight"] = module.weight.detach().clone()
        print(f"  Updated {module_name} | ||delta||={delta.norm().item():.4f}")
    # endregion

    if not edit_dict:
        raise RuntimeError("No FLUX text-side attention weights were edited")
    print(f"Current model status: Edited {target_concepts} into {anchor_concepts or ['null-anchor']}")
    return edit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument("--sd_ckpt", help="base version for FLUX", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    # Erase Config
    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default=None)
    # Hyperparameters
    parser.add_argument("--params", type=str, default="KV", choices=["Q", "K", "V", "QK", "KV", "QKV"])
    parser.add_argument("--threshold", type=float, default=1e-1)
    # FLUX/MEMIT-specific controls
    parser.add_argument("--trace_num_steps", type=int, default=4)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=1e-4)
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
        if len(target_concepts) != len(anchor_concepts):
            raise ValueError("target_concepts and anchor_concepts must have the same length")
        file_suffix += f"-to_{anchor_concepts[0]}_etc"

    retain_texts = []
    if retain_path is not None:
        if not retain_path.endswith(".csv"):
            raise ValueError("--retain_path must be a .csv file")
        if args.heads is None:
            raise ValueError("--heads is required when --retain_path is used")
        df = pd.read_csv(retain_path)
        for head in args.heads.split(","):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    max_sequence_length = 256 if "schnell" in args.sd_ckpt else 512

    pipeline = DiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)
    edit_dict = edit_model(
        args=args,
        pipeline=pipeline,
        target_concepts=target_concepts,
        anchor_concepts=anchor_concepts,
        retain_texts=retain_texts,
        device=args.device,
        max_sequence_length=max_sequence_length,
    )
    os.makedirs(save_path, exist_ok=True)
    save_file(edit_dict, os.path.join(save_path, f"{file_name}.safetensors"))
