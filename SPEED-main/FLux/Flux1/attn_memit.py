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
    else:
        retain_inputs = retain_inputs.to(
            device=target_target.device,
            dtype=target_target.dtype,
        )
        covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]
        U, S, _ = torch.linalg.svd(covariance, full_matrices=False)
        null_basis = U[:, S < retain_threshold]
        projector = eye if null_basis.shape[1] == 0 else null_basis @ null_basis.T

    system = target_target @ projector + update_lambda * eye
    return torch.linalg.solve(
        system.T,
        (residual_target @ projector).T,
    ).T


def _concept_token_indices(pipeline, concepts, max_sequence_length):
    token_indices = {}
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
        token_index = int(token_inputs.attention_mask[0].sum().item()) - 2
        if token_index < 0:
            raise RuntimeError(
                f"Prompt token for {concept!r} was truncated by "
                f"max_sequence_length={max_sequence_length}."
            )
        token_indices[concept] = [token_index]
    return token_indices


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
    selected_suffixes = [ATTENTION_SUFFIXES[param] for param in args.params]
    edit_modules = []
    for name, module in pipeline.transformer.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not any(name.endswith(suffix) for suffix in selected_suffixes):
            continue
        if re.match(r"transformer_blocks\.\d+\.", name) is None:
            continue
        edit_modules.append((name, module))

    if not edit_modules:
        raise RuntimeError(
            f"No Flux1 text-side attention modules selected for params={args.params}"
        )

    module_names = [name for name, _module in edit_modules]
    final_module_name = module_names[-1]
    remaining_counts = {
        name: len(module_names) - index
        for index, name in enumerate(module_names)
    }

    concepts = list(dict.fromkeys(target_concepts + anchor_concepts + retain_texts))
    concept_indices = _concept_token_indices(
        pipeline,
        concepts,
        max_sequence_length,
    )
    target_token_indices = {
        concept: concept_indices[concept]
        for concept in target_concepts
    }
    anchor_token_indices = {
        concept: [0] if concept == "" else concept_indices[concept]
        for concept in anchor_concepts
    }
    retain_token_indices = {
        concept: list(range(1, max_sequence_length))
        if concept == ""
        else concept_indices[concept]
        for concept in retain_texts
    }

    anchor_final_traces = _trace_concepts(
        pipeline,
        anchor_concepts,
        anchor_token_indices,
        [final_module_name],
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
    for module_name, module in edit_modules:
        trace_names = [module_name]
        if final_module_name != module_name:
            trace_names.append(final_module_name)
        target_traces = _trace_concepts(
            pipeline,
            target_concepts,
            target_token_indices,
            trace_names,
            args,
            device,
            max_sequence_length,
        )

        target_inputs = []
        residuals = []
        for concept, anchor_concept in zip(target_concepts, anchor_concepts):
            concept_trace = target_traces[concept]
            current_final = concept_trace[final_module_name]["outputs"]
            anchor_final = anchor_final_traces[anchor_concept][final_module_name][
                "outputs"
            ].to(device=current_final.device, dtype=current_final.dtype)
            target_inputs.append(concept_trace[module_name]["inputs"])
            residuals.append(
                args.residual_scale
                * (anchor_final - current_final)
                / remaining_counts[module_name]
            )

        target_target = torch.stack(
            [target @ target.T for target in target_inputs]
        ).mean(0)
        residual_target = torch.stack(
            [residual @ target.T for target, residual in zip(target_inputs, residuals)]
        ).mean(0)
        target_device = module.weight.device
        delta = _closed_form_update(
            residual_target.to(target_device, torch.float32),
            target_target.to(target_device, torch.float32),
            args.update_lambda,
            retain_inputs_by_module[module_name].to(target_device, torch.float32),
            args.threshold,
        )
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
    parser.add_argument("--trace_num_steps", type=int, default=4)
    parser.add_argument("--trace_guidance_scale", type=float, default=0.0)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--update_lambda", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=1.0)
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
