#!/usr/bin/env python3
"""Build GCD-screened erase or retain celebrity prompt CSVs for FLUX.1-dev."""

import argparse
import csv
import fcntl
import gc
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
FLUX_DIR = SCRIPT_DIR.parent
DATA_DIR = FLUX_DIR / "data"
DEFAULT_GCD_SCRIPT = SCRIPT_DIR / "GCD.py"
DEFAULT_GCD_PYTHONPATH = FLUX_DIR / "celeb-detection-oss"
DEFAULT_LABEL_TABLE = DATA_DIR / "gcd_labels.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "gcd_verified"
DEFAULT_WORK_DIR = FLUX_DIR / "logs" / "gcd_verified"
COLUMNS = ("id", "type", "text", "concept", "seed")
TEMPLATES = (
    "A portrait of {}",
    "An image capturing {} at a public event",
    "An oil painting of {}",
    "A sketch of {}",
    "{} in an official photo",
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    row_type: str
    people: int
    seeds_per_template: int
    max_attempts: int
    minimum_passes: int
    base_start: int
    base_stop: int

    def values(self) -> Tuple[int, int, int, int]:
        return self.people, self.seeds_per_template, self.max_attempts, self.minimum_passes


@dataclass(frozen=True)
class Job:
    prompt: str
    seed: int

    @property
    def filename(self) -> str:
        return f"{self.prompt}_{self.seed}.png"


def task_spec(task: str, erase_count: Optional[int] = None) -> TaskSpec:
    if task == "retain":
        return TaskSpec("retain", "retain", 100, 1, 21, 4, 100, 200)
    if task != "erase" or erase_count not in (10, 50, 100):
        raise ValueError("erase requires --erase-count 10, 50, or 100")
    seeds = {10: 10, 50: 2, 100: 1}[erase_count]
    return TaskSpec(
        f"erase-{erase_count}", "erase", erase_count, seeds,
        21,
        (5 * seeds * 9 + 9) // 10 if erase_count < 100 else 4,
        0, 100,
    )


def progress_label(spec: TaskSpec) -> str:
    if spec.row_type == "erase":
        return f"{spec.people}_celebrity"
    return spec.row_type


def base_candidates(spec: TaskSpec, labels: Sequence[str]) -> List[str]:
    if len(labels) < 200:
        raise ValueError("gcd_labels.csv must contain at least 200 names")
    return list(labels[spec.base_start:spec.base_stop])


def fallback_candidates(labels: Sequence[str]) -> List[str]:
    return list(labels[200:])


def accepted(spec: TaskSpec, counts: Sequence[int], exhausted: bool) -> bool:
    total = sum(min(count, spec.seeds_per_template) for count in counts)
    target = 5 * spec.seeds_per_template
    return total == target or exhausted and total >= spec.minimum_passes


def output_rows(
    concepts: Sequence[str], spec: TaskSpec, passed: Mapping[str, Mapping[str, Sequence[int]]]
) -> List[Dict[str, str]]:
    rows = []
    for concept in concepts:
        for template in TEMPLATES:
            prompt = template.format(concept)
            seeds = list(dict.fromkeys(passed[concept].get(prompt, ())))
            seeds.extend(seed for seed in range(spec.max_attempts) if seed not in seeds)
            for seed in seeds[:spec.seeds_per_template]:
                rows.append({"id": str(len(rows) + 1), "type": spec.row_type, "text": prompt, "concept": concept, "seed": str(seed)})
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"Expected columns {COLUMNS} in {path}")
        return [dict(row) for row in reader]


def saved_concepts(path: Path) -> List[str]:
    return list(dict.fromkeys(row["concept"] for row in read_rows(path)))


def load_labels(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = {name.lower(): name for name in reader.fieldnames or ()}
        column = next((fields[name] for name in ("concept", "label", "labels", "name") if name in fields), None)
        if column is None:
            raise ValueError(f"No label column in {path}")
        return [row[column].split("_[", 1)[0].replace("_", " ").strip() for row in reader if row.get(column, "").strip()]


def output_path(spec: TaskSpec, output_dir: Path) -> Path:
    name = "verified_retain.csv" if spec.row_type == "retain" else f"verified_{spec.people}_celebrity_erase.csv"
    return output_dir / name


def passed_path(spec: TaskSpec, output_dir: Path) -> Path:
    path = output_path(spec, output_dir)
    return path.with_name(f"{path.stem}_gcd_passed.csv")


def load_passed(path: Path) -> Dict[str, Dict[str, List[int]]]:
    result: Dict[str, Dict[str, List[int]]] = {}
    for row in read_rows(path):
        seeds = result.setdefault(row["concept"], {}).setdefault(row["text"], [])
        seed = int(row["seed"])
        if seed not in seeds:
            seeds.append(seed)
    return result


def passed_rows(spec: TaskSpec, passed: Mapping[str, Mapping[str, Sequence[int]]]) -> List[Dict[str, str]]:
    rows = []
    for concept, prompts in passed.items():
        for prompt, seeds in prompts.items():
            for seed in dict.fromkeys(seeds):
                rows.append({"id": str(len(rows) + 1), "type": spec.row_type, "text": prompt, "concept": concept, "seed": str(seed)})
    return rows


def claim_fallback(registry: Path, task_name: str, labels: Sequence[str]) -> str:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        used = {row["concept"] for row in csv.DictReader(handle)}
        concept = next((name for name in fallback_candidates(labels) if name not in used), None)
        if concept is None:
            raise RuntimeError("No unused labels remain after label 200")
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=("task", "concept"))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({"task": task_name, "concept": concept})
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return concept


def gcd_environment(gcd_pythonpath: Optional[Path]) -> Dict[str, str]:
    env = os.environ.copy()
    if gcd_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(part for part in (str(gcd_pythonpath), env.get("PYTHONPATH")) if part)
    return env


def read_gcd_results(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: dict(row) for row in csv.DictReader(handle)}


def run_gcd(
    script: Path, image_dir: Path, results_path: Path, python: str, pythonpath: Optional[Path]
) -> Dict[str, Dict[str, str]]:
    result = subprocess.run(
        [python, str(script), "--image_folder", str(image_dir), "--results-csv", str(results_path)],
        capture_output=True, text=True, check=False, env=gcd_environment(pythonpath),
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode:
        raise RuntimeError(f"GCD exited with {result.returncode}:\n{output}")
    return read_gcd_results(results_path)


def load_pipeline(model_id: str, device: str) -> Any:
    import torch
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    return pipe


def generate(pipe: Any, jobs: Sequence[Job], image_dir: Path, args: argparse.Namespace) -> None:
    import torch
    image_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        image = pipe(prompt=job.prompt, generator=torch.Generator(args.generation_device).manual_seed(job.seed), num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, height=args.height, width=args.width, max_sequence_length=args.max_sequence_length).images[0]
        image.save(image_dir / job.filename)


def verify_concept(pipe: Any, concept: str, spec: TaskSpec, known: Dict[str, List[int]], args: argparse.Namespace) -> bool:
    prompts = [template.format(concept) for template in TEMPLATES]
    template_numbers = {prompt: index for index, prompt in enumerate(prompts, start=1)}
    label = progress_label(spec)
    passed = {prompt: list(dict.fromkeys(known.get(prompt, ())))[:spec.seeds_per_template] for prompt in prompts}
    attempted = {prompt: set(passed[prompt]) for prompt in prompts}
    batch = 0
    while True:
        counts = [len(passed[prompt]) for prompt in prompts]
        exhausted = all(len(attempted[prompt]) == spec.max_attempts for prompt in prompts if len(passed[prompt]) < spec.seeds_per_template)
        if accepted(spec, counts, exhausted):
            known.update(passed)
            return True
        if exhausted:
            return False
        jobs = []
        for prompt in prompts:
            if len(passed[prompt]) >= spec.seeds_per_template or len(attempted[prompt]) >= spec.max_attempts:
                continue
            seed = next(seed for seed in range(spec.max_attempts) if seed not in attempted[prompt])
            attempted[prompt].add(seed)
            jobs.append(Job(prompt, seed))
        batch += 1
        run_name = f"{spec.name}-{re.sub('[^A-Za-z0-9]+', '_', concept)}-{batch:03d}"
        image_dir = args.work_dir / spec.name / "images" / run_name
        results_path = args.work_dir / spec.name / "gcd-results" / f"{run_name}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            for job in jobs:
                print(
                    f"[{label}] {concept} | template {template_numbers[job.prompt]}/{len(TEMPLATES)} | seed {job.seed} | generating",
                    flush=True,
                )
            generate(pipe, jobs, image_dir, args)
            print(f"[{label}] {concept} | batch {batch} | running GCD", flush=True)
            results = run_gcd(args.gcd_script, image_dir, results_path, args.gcd_python, args.gcd_pythonpath)
            for job in jobs:
                result = results.get(job.filename, {})
                passed_result = result.get("face_detected") == "1" and result.get("correct") == "1"
                top1 = result.get("top1_name") or "no face"
                if passed_result:
                    passed[job.prompt].append(job.seed)
                    retained_dir = args.output_dir / "images" / spec.name / concept
                    retained_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_dir / job.filename, retained_dir / job.filename)
                    print(
                        f"[{label}] {concept} | seed {job.seed} | retained {retained_dir / job.filename}",
                        flush=True,
                    )
                print(
                    f"[{label}] {concept} | template {template_numbers[job.prompt]}/{len(TEMPLATES)} | seed {job.seed} | GCD top-1: {top1} | {'pass' if passed_result else 'retry'}",
                    flush=True,
                )
        finally:
            shutil.rmtree(image_dir, ignore_errors=True)


def completed(spec: TaskSpec, concept: str, passed: Mapping[str, Mapping[str, Sequence[int]]]) -> bool:
    prompts = passed.get(concept, {})
    counts = [len(prompts.get(template.format(concept), ())) for template in TEMPLATES]
    return accepted(spec, counts, exhausted=True)


def run_task(pipe: Any, spec: TaskSpec, labels: Sequence[str], args: argparse.Namespace) -> Path:
    destination = output_path(spec, args.output_dir)
    state = passed_path(spec, args.output_dir)
    registry = args.output_dir / "fallback_claims.csv"
    if args.overwrite:
        destination.unlink(missing_ok=True)
        state.unlink(missing_ok=True)
    passed = load_passed(state)
    selected = [concept for concept in base_candidates(spec, labels) if completed(spec, concept, passed)]
    selected = selected[:spec.people]
    candidates = iter(base_candidates(spec, labels))
    while len(selected) < spec.people:
        try:
            concept = next(candidates)
        except StopIteration:
            concept = claim_fallback(registry, spec.name, labels)
        if concept in selected or completed(spec, concept, passed):
            if concept not in selected:
                selected.append(concept)
            continue
        print(f"[{progress_label(spec)}] testing {concept}", flush=True)
        person_passed: Dict[str, List[int]] = {}
        if verify_concept(pipe, concept, spec, person_passed, args):
            passed[concept] = person_passed
            selected.append(concept)
            write_rows(state, passed_rows(spec, passed))
            write_rows(destination, output_rows(selected, spec, passed))
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("erase", "retain"), required=True)
    parser.add_argument("--erase-count", type=int, choices=(10, 50, 100))
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--generation-device", default="cuda:0")
    parser.add_argument("--gcd-script", type=Path, default=DEFAULT_GCD_SCRIPT)
    parser.add_argument("--gcd-python", default=sys.executable)
    parser.add_argument("--gcd-pythonpath", type=Path, default=DEFAULT_GCD_PYTHONPATH)
    parser.add_argument("--label-table", type=Path, default=DEFAULT_LABEL_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.task == "erase" and args.erase_count is None:
        parser.error("--erase-count is required for erase")
    if not args.gcd_script.is_file() or not args.label_table.is_file():
        parser.error("GCD script or label table does not exist")
    return args


def main() -> None:
    args = parse_args()
    spec = task_spec(args.task, args.erase_count)
    pipe = load_pipeline(args.model_id, args.generation_device)
    try:
        path = run_task(pipe, spec, load_labels(args.label_table), args)
        print(f"wrote {path}", flush=True)
    finally:
        del pipe
        gc.collect()


if __name__ == "__main__":
    main()
