"""Measure Flux1 retain covariance singular values without applying an edit.

The retain analysis uses every position in the configured sequence, including
padding positions. With the default ``max_sequence_length=512``, each retain
prompt contributes all 512 token positions at every traced diffusion step. The
retain covariance matches ``mlp.py`` by dividing the accumulated Gram matrix by
the number of retain concepts, not by the number of traced token columns.
"""

import argparse
import csv
import math
import re
from pathlib import Path


FLUX1_MLP_SUFFIX = ".ff_context.net.2"
DEFAULT_THRESHOLDS = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1)
CSV_FIELDS = (
    "module",
    "layer",
    "matrix_dim",
    "sample_columns",
    "concept_count",
    "threshold",
    "count_below_threshold",
    "null_space_dim",
    "exact_zero_count",
    "numerical_rank_exact",
    "min_singular_value",
    "max_singular_value",
)


def _threshold_label(threshold):
    return format(float(threshold), ".12g")


def summarize_singular_values(singular_values, thresholds):
    """Return exact-zero and strict-below-threshold counts.

    ``singular_values`` may be a PyTorch tensor or a regular iterable, which
    keeps this function usable in dependency-free unit tests.
    """
    if hasattr(singular_values, "detach"):
        values = singular_values.detach().cpu().reshape(-1).tolist()
    else:
        values = [float(value) for value in singular_values]
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("singular_values contains a non-finite value")

    parsed_thresholds = [float(threshold) for threshold in thresholds]
    if any(not math.isfinite(threshold) or threshold < 0 for threshold in parsed_thresholds):
        raise ValueError("thresholds must be finite and non-negative")

    summary = {
        "exact_zero_count": sum(value == 0.0 for value in values),
        "numerical_rank_exact": sum(value != 0.0 for value in values),
        "min_singular_value": min(values) if values else float("nan"),
        "max_singular_value": max(values) if values else float("nan"),
    }
    for threshold in parsed_thresholds:
        summary[f"count_below_{_threshold_label(threshold)}"] = sum(
            value < threshold for value in values
        )
    return summary


def write_summary_csv(rows, output_path):
    """Write one row per module and threshold."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_retain_texts(retain_path, heads, target_concepts, retain_type="retain"):
    if retain_path is None:
        return [""]
    with Path(retain_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        columns = [head.strip() for head in heads.split(",") if head.strip()]
        missing = [column for column in columns if column not in fieldnames]
        if missing:
            raise KeyError(f"Retain CSV has no column(s) {', '.join(missing)!r}")
        use_type_filter = bool(retain_type) and "type" in fieldnames
        texts = []
        for row in reader:
            if use_type_filter and row.get("type", "").strip().lower() != retain_type.lower():
                continue
            texts.extend(row[column] for column in columns if row.get(column))

    patterns = [
        re.compile(r"\b" + re.escape(concept.lower()) + r"\b")
        for concept in target_concepts
        if concept
    ]
    return list(dict.fromkeys(
        text for text in texts
        if not any(pattern.search(text.lower()) for pattern in patterns)
    ))


def _select_mlp_modules(pipeline):
    modules = []
    for layer_index, block in enumerate(pipeline.transformer.transformer_blocks):
        name = f"transformer_blocks.{layer_index}{FLUX1_MLP_SUFFIX}"
        modules.append((name, block.ff_context.net[2]))
    if not modules:
        raise RuntimeError("No Flux1 text MLP modules found")
    return modules


def _retain_token_indices(pipeline, retain_texts, max_sequence_length):
    all_positions = list(range(max_sequence_length))
    return {text: all_positions.copy() for text in retain_texts}


def analyze_retain_set(
    pipeline,
    retain_texts,
    args,
    device,
    max_sequence_length,
    thresholds,
    svd_device="cpu",
    svd_dtype="float64",
):
    """Trace retain prompts and return per-module singular-value summaries."""
    import torch
    try:
        from mlp import _trace_concepts
    except ModuleNotFoundError:
        import importlib.util

        mlp_path = Path(__file__).with_name("mlp.py")
        spec = importlib.util.spec_from_file_location("flux1_mlp", mlp_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Flux1 trace implementation from {mlp_path}")
        mlp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mlp_module)
        _trace_concepts = mlp_module._trace_concepts

    modules = _select_mlp_modules(pipeline)
    module_names = [name for name, _module in modules]
    if not retain_texts:
        raise ValueError("No retain prompts remain after CSV loading and target filtering")
    token_indices = _retain_token_indices(pipeline, retain_texts, max_sequence_length)
    second_moment = {name: None for name in module_names}
    sample_columns = {name: 0 for name in module_names}
    concept_count = {name: 0 for name in module_names}

    def accumulate(_concept, concept_trace):
        for name in module_names:
            inputs = concept_trace[name]["inputs"]
            gram = inputs @ inputs.T
            if second_moment[name] is None:
                second_moment[name] = gram
            else:
                second_moment[name].add_(gram)
            sample_columns[name] += inputs.shape[1]
            concept_count[name] += 1
            del inputs, gram

    _trace_concepts(
        pipeline,
        retain_texts,
        token_indices,
        module_names,
        args,
        device,
        max_sequence_length,
        on_concept_trace=accumulate,
    )

    rows = []
    torch_dtype = torch.float64 if svd_dtype == "float64" else torch.float32
    for layer, (name, _module) in enumerate(modules):
        count = concept_count[name]
        if second_moment[name] is None or count == 0:
            raise RuntimeError(f"No retain trace for {name}")
        covariance = second_moment[name] / count
        covariance = covariance.to(device=svd_device, dtype=torch_dtype)
        singular_values = torch.linalg.svdvals(covariance)
        summary = summarize_singular_values(singular_values, thresholds)
        for threshold in thresholds:
            threshold = float(threshold)
            rows.append({
                "module": name,
                "layer": layer,
                "matrix_dim": covariance.shape[0],
                "sample_columns": sample_columns[name],
                "concept_count": count,
                "threshold": threshold,
                "count_below_threshold": summary[f"count_below_{_threshold_label(threshold)}"],
                "null_space_dim": summary[f"count_below_{_threshold_label(threshold)}"],
                "exact_zero_count": summary["exact_zero_count"],
                "numerical_rank_exact": summary["numerical_rank_exact"],
                "min_singular_value": summary["min_singular_value"],
                "max_singular_value": summary["max_singular_value"],
            })
        print(
            f"{name}: dim={covariance.shape[0]} columns={sample_columns[name]} "
            f"concepts={count} "
            f"exact_zero={summary['exact_zero_count']} "
            f"min={summary['min_singular_value']:.6e} "
            f"max={summary['max_singular_value']:.6e}"
        )
        for threshold in thresholds:
            label = _threshold_label(threshold)
            null_dim = summary[f"count_below_{label}"]
            print(f"  S < {label}: {null_dim} (null_space_dim={null_dim})")
        del covariance, singular_values
    return rows


def _parse_thresholds(value):
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")
    if any(not math.isfinite(item) or item < 0 for item in thresholds):
        raise ValueError("--thresholds must be finite and non-negative")
    return list(dict.fromkeys(thresholds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd_ckpt", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--retain_path", default=None)
    parser.add_argument("--heads", default="concept")
    parser.add_argument("--retain_type", default="retain")
    parser.add_argument("--target_concepts", default="")
    parser.add_argument("--output_csv", default="logs/retain_singular_values.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trace_num_steps", type=int, default=10)
    parser.add_argument("--trace_seed", type=int, default=0)
    parser.add_argument("--trace_resolution", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated strict-below thresholds for singular-value counts",
    )
    parser.add_argument("--svd_device", default=None)
    parser.add_argument("--svd_dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    target_concepts = [item.strip() for item in args.target_concepts.split(",") if item.strip()]
    retain_texts = _load_retain_texts(
        args.retain_path, args.heads, target_concepts, args.retain_type
    )
    thresholds = _parse_thresholds(args.thresholds)
    pipeline = DiffusionPipeline.from_pretrained(
        args.sd_ckpt, torch_dtype=torch.float32
    ).to(args.device)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    rows = analyze_retain_set(
        pipeline,
        retain_texts,
        args,
        args.device,
        args.max_sequence_length,
        thresholds,
        args.svd_device or args.device,
        args.svd_dtype,
    )
    write_summary_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
