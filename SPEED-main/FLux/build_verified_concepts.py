#!/usr/bin/env python3
"""Generate FLUX2 celebrity images and retain GCD-verified concepts."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONCEPT_TABLE = SCRIPT_DIR / "data" / "vggface.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "verified_celebrity.csv"
DEFAULT_WORK_DIR = SCRIPT_DIR / "logs" / "gcd_celebrity_images"
DEFAULT_GCD_PYTHONPATH = SCRIPT_DIR / "celeb-detection-oss"
SAMPLES_PER_CONCEPT = 5
GCD_CELEBRITY_TEMPLATES = (
    "A portrait of {}",
    "An image capturing {} at a public event",
    "An oil painting of {}",
    "A sketch of {}",
    "{} in an official photo",
)
DEFAULT_GCD_ACCURACY_PATTERN = (
    r"(?im)\b(?:gcd\s*)?(?:accuracy|acc)\b\s*(?:is\s*)?"
    r"[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
)


@dataclass(frozen=True)
class SampleJob:
    index: int
    prompt: str
    seed: int
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retain celebrity concepts whose FLUX2 outputs pass an external GCD check."
    )
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--generation-device", default="cuda:0")
    parser.add_argument("--gcd-script", type=Path, required=True)
    parser.add_argument(
        "--gcd-python",
        default=sys.executable,
        help="Python interpreter for the GCD environment (defaults to this interpreter).",
    )
    parser.add_argument(
        "--gcd-pythonpath",
        type=Path,
        default=DEFAULT_GCD_PYTHONPATH,
        help="Directory containing celeb-detection-oss model_training package.",
    )
    parser.add_argument("--gcd-accuracy-pattern", default=DEFAULT_GCD_ACCURACY_PATTERN)
    parser.add_argument("--concept-table", type=Path, default=DEFAULT_CONCEPT_TABLE)
    parser.add_argument("--concept-count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--minimum-gcd-accuracy", type=float, default=0.60)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--limit-concepts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.gcd_script.is_file():
        parser.error(f"GCD script does not exist: {args.gcd_script}")
    if not 0.0 <= args.minimum_gcd_accuracy <= 1.0:
        parser.error("--minimum-gcd-accuracy must be between 0 and 1")
    if args.limit_concepts is not None and args.limit_concepts <= 0:
        parser.error("--limit-concepts must be positive")
    if args.concept_count <= 0:
        parser.error("--concept-count must be positive")
    return args


def load_candidate_concepts(path: Path, limit: int | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "concept" not in reader.fieldnames:
            raise ValueError(f"Source CSV must contain a 'concept' column: {path}")
        concepts = []
        seen = set()
        for row in reader:
            concept = (row.get("concept") or "").strip()
            if concept and concept not in seen:
                concepts.append(concept)
                seen.add(concept)
    return concepts if limit is None else concepts[:limit]


def select_candidate_concepts(labels: Iterable[str], concept_count: int) -> list[str]:
    concepts = []
    seen = set()
    for label in labels:
        concept = str(label).strip()
        if concept and concept not in seen:
            concepts.append(concept)
            seen.add(concept)
        if len(concepts) == concept_count:
            break
    if len(concepts) < concept_count:
        return concepts
    return concepts


def build_sampling_jobs(concept: str, samples_per_concept: int, base_seed: int) -> list[SampleJob]:
    jobs = []
    template_count = len(GCD_CELEBRITY_TEMPLATES)
    for index in range(samples_per_concept):
        template = GCD_CELEBRITY_TEMPLATES[index % template_count]
        sample_round = index // template_count
        prompt = template.format(concept)
        jobs.append(
            SampleJob(
                index=index,
                prompt=prompt,
                seed=base_seed + index,
                filename=f"{prompt}_{sample_round}.png",
            )
        )
    return jobs


def extract_gcd_accuracy(output: str, pattern: str = DEFAULT_GCD_ACCURACY_PATTERN) -> float:
    match = re.search(pattern, output)
    if match is None:
        raise ValueError(
            "Could not find GCD accuracy in evaluator output. "
            f"Expected a value matching: {pattern!r}"
        )
    accuracy = float(match.group(1))
    if accuracy > 1.0:
        accuracy /= 100.0
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"GCD accuracy must be in [0, 1] or [0, 100], got {accuracy}")
    return accuracy


def is_gcd_verified(accuracy: float, minimum_accuracy: float) -> bool:
    return accuracy >= minimum_accuracy


def run_gcd(
    gcd_script: Path,
    image_dir: Path,
    accuracy_pattern: str,
    gcd_python: str = sys.executable,
    gcd_pythonpath: Path | None = None,
) -> tuple[float, str]:
    env = os.environ.copy()
    if gcd_pythonpath is not None:
        pythonpath_parts = [str(gcd_pythonpath)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    result = subprocess.run(
        [gcd_python, str(gcd_script), "--image_folder", str(image_dir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise RuntimeError(
            f"GCD evaluator failed with exit code {result.returncode}:\n{output}"
        )
    return extract_gcd_accuracy(output, accuracy_pattern), output


def write_gcd_result(
    path: Path,
    concept: str,
    accuracy: float,
    minimum_accuracy: float,
    output: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = is_gcd_verified(accuracy, minimum_accuracy)
    path.write_text(
        f"concept: {concept}\n"
        f"gcd_accuracy: {accuracy:.6f}\n"
        f"minimum_gcd_accuracy: {minimum_accuracy:.6f}\n"
        f"passed: {passed}\n"
        "--- evaluator output ---\n"
        f"{output}\n",
        encoding="utf-8",
    )


def write_verified_concepts(path: Path, concepts: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "concept"])
        for index, concept in enumerate(concepts, start=1):
            writer.writerow([index, concept])


def read_verified_concepts(path: Path) -> list[str]:
    return load_candidate_concepts(path) if path.is_file() else []


def load_generation_pipeline(model_id: str, device: str):
    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
    try:
        pipeline.set_progress_bar_config(disable=True)
    except AttributeError:
        pass
    return pipeline


def generate_images(pipeline: Any, jobs: list[SampleJob], output_dir: Path, args: argparse.Namespace) -> list[Path]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for job in jobs:
        generator = torch.Generator(device=args.generation_device).manual_seed(job.seed)
        result = pipeline(
            prompt=job.prompt,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
        )
        image_path = output_dir / job.filename
        result.images[0].save(image_path)
        image_paths.append(image_path)
    return image_paths


def process_concepts(
    pipeline: Any,
    concepts: list[str],
    args: argparse.Namespace,
) -> None:
    verified = [] if args.overwrite else read_verified_concepts(args.output)
    verified_set = set(verified)
    if args.limit_concepts is not None:
        concepts = concepts[:args.limit_concepts]
    for concept_index, concept in enumerate(concepts):
        if concept in verified_set:
            print(f"skipping already verified concept: {concept}")
            continue
        jobs = build_sampling_jobs(
            concept=concept,
            samples_per_concept=SAMPLES_PER_CONCEPT,
            base_seed=args.base_seed + concept_index * SAMPLES_PER_CONCEPT,
        )
        concept_dir = args.work_dir / f"{concept_index:04d}"
        if concept_dir.exists():
            shutil.rmtree(concept_dir)
        print(f"{concept_index + 1}/{len(concepts)}: generating {SAMPLES_PER_CONCEPT} images for {concept}")
        try:
            image_paths = generate_images(pipeline, jobs, concept_dir, args)
            if len(image_paths) != SAMPLES_PER_CONCEPT:
                raise RuntimeError(
                    f"Expected {SAMPLES_PER_CONCEPT} generated images, got {len(image_paths)}"
                )
            gcd_accuracy, gcd_output = run_gcd(
                args.gcd_script,
                concept_dir,
                args.gcd_accuracy_pattern,
                args.gcd_python,
                args.gcd_pythonpath,
            )
            gcd_result_path = args.work_dir / f"{concept_index:04d}.gcd.txt"
            write_gcd_result(
                gcd_result_path,
                concept,
                gcd_accuracy,
                args.minimum_gcd_accuracy,
                gcd_output,
            )
            passed = is_gcd_verified(gcd_accuracy, args.minimum_gcd_accuracy)
            print(
                f"  GCD accuracy: {gcd_accuracy:.2%} "
                f"({'accept' if passed else 'reject'}; "
                f"minimum {args.minimum_gcd_accuracy:.2%})"
            )
            if passed:
                verified.append(concept)
                verified_set.add(concept)
                write_verified_concepts(args.output, verified)
                print(f"kept {concept}: GCD accuracy {gcd_accuracy:.2%}")
            else:
                print(f"rejected {concept}: GCD accuracy {gcd_accuracy:.2%}")
        finally:
            shutil.rmtree(concept_dir, ignore_errors=True)
    write_verified_concepts(args.output, verified)


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output.exists():
        args.output.unlink()
    print("Loading FLUX.2 Klein 4B generation pipeline...")
    pipeline = load_generation_pipeline(args.model_id, args.generation_device)
    print(
        f"Using GCD evaluator {args.gcd_script} with "
        f"minimum accuracy {args.minimum_gcd_accuracy:.2%}"
    )
    try:
        candidate_concepts = load_candidate_concepts(args.concept_table)
        concepts = select_candidate_concepts(candidate_concepts, args.concept_count)
        print(f"Loaded {len(concepts)} candidate concepts from {args.concept_table}")
        process_concepts(
            pipeline,
            concepts,
            args,
        )
    finally:
        del pipeline
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
