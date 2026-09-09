import os
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import time

import pandas as pd
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from safetensors.torch import save_file


ATTENTION_SUFFIXES = {
    "Q": ".attn.add_q_proj",
    "K": ".attn.add_k_proj",
    "V": ".attn.add_v_proj",
}
FLUX1_MAX_SEQUENCE_LENGTH = 256


def _trace_concepts(
    pipeline,
    concepts,
    token_indices,
    module_names,
    args,
    device,
    max_sequence_length,
):
    module_lookup = dict(pipeline.transformer.named_modules())
    traced_concepts = {}
    trace_batch_size = max(1, int(getattr(args, "trace_batch_size", 1)))
    grouped_concepts = {}

    for concept in dict.fromkeys(concepts):
        selected_indices = token_indices.get(concept)
        if selected_indices:
            grouped_concepts.setdefault(tuple(selected_indices), []).append(concept)

    for selected_indices, grouped in grouped_concepts.items():
        selected_indices = list(selected_indices)
        for start in range(0, len(grouped), trace_batch_size):
            concept_batch = grouped[start : start + trace_batch_size]
            traces = {name: {"inputs": [], "outputs": []} for name in module_names}
            handles = []

            for name in module_names:
                module = module_lookup[name]

                def pre_hook(_module, inputs, module_name=name):
                    values = inputs[0][:, selected_indices, :]
                    traces[module_name]["inputs"].append(values.detach().float())

                def output_hook(_module, _inputs, output, module_name=name):
                    values = output[0] if isinstance(output, tuple) else output
                    values = values[:, selected_indices, :]
                    traces[module_name]["outputs"].append(values.detach().float())

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(output_hook))

            generators = [
                torch.Generator(device=device).manual_seed(args.trace_seed)
                for _ in concept_batch
            ]
            try:
                with torch.no_grad():
                    pipeline(
                        prompt=concept_batch,
                        generator=generators,
                        num_inference_steps=args.trace_num_steps,
                        guidance_scale=getattr(args, "trace_guidance_scale", 0.0),
                        height=args.trace_resolution,
                        width=args.trace_resolution,
                        max_sequence_length=max_sequence_length,
                        output_type="latent",
                    )
            finally:
                for handle in handles:
                    handle.remove()

            for batch_index, concept in enumerate(concept_batch):
                compact = {}
                for name, record in traces.items():
                    compact[name] = {}
                    for kind in ("inputs", "outputs"):
                        stacked = torch.stack(record[kind], dim=0)
                        compact[name][kind] = stacked[:, batch_index, :, :].reshape(
                            -1, stacked.shape[-1]
                        ).T
                traced_concepts[concept] = compact

    return traced_concepts


def _closed_form_update(
    residual_target,
    target_target,
    update_lambda,
    retain_inputs,
    retain_threshold,
):
    eye = torch.eye(
        target_target.shape[0],
        device=target_target.device,
        dtype=target_target.dtype,
    )
    if retain_inputs is None:
        projector = eye
        projector_rank = projector.shape[0]
    else:
        retain_inputs = retain_inputs.to(
            device=target_target.device,
            dtype=target_target.dtype,
        )
        covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
        U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
        null_basis = U[:, S < retain_threshold]
        projector = eye if null_basis.shape[1] == 0 else null_basis @ null_basis.T
        projector_rank = projector.shape[0] if null_basis.shape[1] == 0 else null_basis.shape[1]

    system = target_target @ projector + update_lambda * eye
    delta = torch.linalg.solve(
        system.T,
        (residual_target @ projector).T,
    ).T
    return delta, projector, projector_rank


def _concept_token_indices(pipeline, concepts, max_sequence_length):
    token_indices = {}
    all_token_indices = {}
    for concept in dict.fromkeys(concepts):
        if concept == "":
            continue
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
        token_indices[concept] = content_indices
        all_token_indices[concept] = list(range(valid_token_count))
    return token_indices, all_token_indices


def _load_retain_texts(args, target_concepts):
    if args.retain_path is None:
        retain_texts = [""]
    else:
        if not args.retain_path.endswith(".csv"):
            raise ValueError("--retain_path must point to a CSV file")
        if not args.heads:
            raise ValueError("--heads is required when --retain_path is provided")
        dataframe = pd.read_csv(args.retain_path)
        retain_texts = []
        for head in args.heads.split(","):
            retain_texts.extend(dataframe[head.strip()].unique().tolist())

    return [
        str(text)
        for text in retain_texts
        if not any(
            re.search(r"\b" + re.escape(concept.lower()) + r"\b", str(text).lower())
            for concept in target_concepts
        )
    ]


def edit_model(
    args,
    pipeline,
    target_concepts,
    anchor_concepts,
    retain_texts,
    device="cuda:0",
    max_sequence_length=FLUX1_MAX_SEQUENCE_LENGTH,
):
    selected_params = list(dict.fromkeys(args.params))
    edit_modules = []
    for name, module in pipeline.transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        matched_params = [
            param
            for param in selected_params
            if name.endswith(ATTENTION_SUFFIXES[param])
        ]
        if not matched_params:
            continue
        if re.match(r"transformer_blocks\.\d+\.", name) is None:
            continue
        edit_modules.append((name, module, matched_params[0]))

    if not edit_modules:
        raise RuntimeError(
            f"No Flux1 text-side attention modules selected for params={args.params}"
        )

    module_names = [name for name, _module, _param in edit_modules]
    module_by_name = {
        module_name: module
        for module_name, module, _param in edit_modules
    }
    modules_by_param = {param: [] for param in selected_params}
    modules_by_layer = {}
    for module_name, module, param in edit_modules:
        modules_by_param[param].append((module_name, module))
        layer_index = int(
            re.match(r"transformer_blocks\.(\d+)\.", module_name).group(1)
        )
        modules_by_layer.setdefault(layer_index, []).append(
            (module_name, module, param)
        )
    if any(not modules for modules in modules_by_param.values()):
        missing = [param for param, modules in modules_by_param.items() if not modules]
        raise RuntimeError(f"No selected Flux1 attention modules found for params={missing}")

    final_module_by_param = {
        param: modules[-1][0]
        for param, modules in modules_by_param.items()
    }
    remaining_counts_by_param = {
        param: {
            name: len(modules) - index
            for index, (name, _module) in enumerate(modules)
        }
        for param, modules in modules_by_param.items()
    }

    concepts = list(dict.fromkeys(target_concepts + anchor_concepts + retain_texts))
    concept_indices, concept_all_token_indices = _concept_token_indices(
        pipeline,
        concepts,
        max_sequence_length,
    )
    target_token_indices = {
        concept: concept_all_token_indices[concept]
        for concept in target_concepts
    }
    anchor_token_indices = {
        concept: [0] if concept == "" else [concept_indices[concept][-1]]
        for concept in anchor_concepts
    }
    retain_token_indices = {
        concept: list(range(1, max_sequence_length))
        if concept == ""
        else [concept_indices[concept][-1]]
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

    retain_inputs_by_module = {name: [] for name in module_names}
    for start in range(0, len(retain_texts), args.chunk_size):
        retain_chunk = retain_texts[start : start + args.chunk_size]
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
                retain_traces[concept][module_name]["inputs"]
                for concept in retain_chunk
                if concept in retain_traces and module_name in retain_traces[concept]
            ]
            if inputs:
                retain_inputs_by_module[module_name].append(torch.cat(inputs, dim=1))

    for module_name in module_names:
        if not retain_inputs_by_module[module_name]:
            raise RuntimeError(f"No retain trace for {module_name}")
        retain_inputs_by_module[module_name] = torch.cat(
            retain_inputs_by_module[module_name],
            dim=1,
        )

    edit_dict = {}
    for layer_index in sorted(modules_by_layer):
        layer_modules = modules_by_layer[layer_index]
        trace_names = list(
            dict.fromkeys(
                [module_name for module_name, _module, _param in layer_modules]
                + [
                    final_module_by_param[param]
                    for _module_name, _module, param in layer_modules
                ]
            )
        )
        target_traces = _trace_concepts(
            pipeline,
            target_concepts,
            target_token_indices,
            trace_names,
            args,
            device,
            max_sequence_length,
        )

        layer_deltas = {}
        for module_name, module, param in layer_modules:
            final_module_name = final_module_by_param[param]
            target_inputs = []
            residuals = []
            for concept, anchor_concept in zip(target_concepts, anchor_concepts):
                concept_trace = target_traces[concept]
                current_final = concept_trace[final_module_name]["outputs"]
                anchor_final = anchor_final_traces[anchor_concept][final_module_name][
                    "outputs"
                ].to(device=current_final.device, dtype=current_final.dtype)
                if current_final.shape[1] % anchor_final.shape[1] != 0:
                    raise RuntimeError(
                        f"Trace shape mismatch: target={current_final.shape}, "
                        f"anchor={anchor_final.shape}"
                    )
                target_count = current_final.shape[1] // anchor_final.shape[1]
                anchor_final = anchor_final.repeat_interleave(target_count, dim=1)
                target_inputs.append(concept_trace[module_name]["inputs"])
                residuals.append(
                    args.residual_scale
                    * (anchor_final - current_final)
                    / remaining_counts_by_param[param][module_name]
                )

            target_inputs = torch.cat(target_inputs, dim=1)
            residuals = torch.cat(residuals, dim=1)
            target_target = target_inputs @ target_inputs.T
            residual_target = residuals @ target_inputs.T
            effective_update_lambda = args.update_lambda * target_inputs.shape[1]
            target_device = module.weight.device
            weight_before = module.weight.detach().float()
            delta, projector, projector_rank = _closed_form_update(
                residual_target.to(target_device, torch.float32),
                target_target.to(target_device, torch.float32),
                effective_update_lambda,
                retain_inputs_by_module[module_name].to(target_device, torch.float32),
                args.threshold,
            )
            projected_inputs = projector @ target_inputs.to(target_device, torch.float32)
            residual_target_values = residuals.to(target_device, torch.float32)
            projected_fit = torch.linalg.vector_norm(
                delta @ projected_inputs - residual_target_values
            ) / torch.linalg.vector_norm(residual_target_values).clamp_min(1e-12)
            relative_update = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
                weight_before.to(target_device, torch.float32)
            ).clamp_min(1e-12)
            print(
                f"[{param}] {module_name}: rel_update={relative_update.item():.3e}, "
                f"projected_fit={projected_fit.item():.3e}, "
                f"retain_projector_rank={projector_rank}, "
                f"effective_lambda={effective_update_lambda:.3e}"
            )
            layer_deltas[module_name] = delta.detach().clone()

        for module_name, delta in layer_deltas.items():
            module = module_by_name[module_name]
            module.weight = torch.nn.Parameter(
                module.weight.float().add(delta).to(module.weight.dtype)
            )
            edit_dict[module_name + ".weight"] = module.weight.detach().clone()

    return edit_dict


def _parse_concepts(value, argument_name):
    concepts = [concept.strip() for concept in value.split(",")]
    if not concepts or any(concept == "" for concept in concepts):
        raise ValueError(f"{argument_name} must not contain empty concepts")
    return concepts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--file_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--trace_batch_size", type=int, default=4)
    parser.add_argument(
        "--params",
        type=str,
        default="KV",
        choices=["Q", "K", "V", "QK", "KV", "QKV"],
    )
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--trace_num_steps", type=int, default=20)
    parser.add_argument("--trace_guidance_scale", type=float, default=3.5)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=6)
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=FLUX1_MAX_SEQUENCE_LENGTH,
    )
    args = parser.parse_args()

    diffusers_logging.set_verbosity_error()
    diffusers_logging.disable_progress_bar()

    target_concepts = _parse_concepts(args.target_concepts, "--target_concepts")
    anchor_concepts = [concept.strip() for concept in args.anchor_concepts.split(",")]
    if len(anchor_concepts) == 1:
        anchor_concepts *= len(target_concepts)
    elif len(anchor_concepts) != len(target_concepts):
        raise ValueError("--anchor_concepts must contain one value or match target count")

    retain_texts = _load_retain_texts(args, target_concepts)
    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}-attn-memit"
    file_suffix += "-to_null" if anchor_concepts[0] == "" else f"-to_{anchor_concepts[0]}"

    pipeline = DiffusionPipeline.from_pretrained(
        args.sd_ckpt,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    pipeline.set_progress_bar_config(disable=True)
    edit_dict = edit_model(
        args=args,
        pipeline=pipeline,
        target_concepts=target_concepts,
        anchor_concepts=anchor_concepts,
        retain_texts=retain_texts,
        device=args.device,
        max_sequence_length=args.max_sequence_length,
    )

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    save_file(edit_dict, os.path.join(save_path, f"{file_name}.safetensors"))
