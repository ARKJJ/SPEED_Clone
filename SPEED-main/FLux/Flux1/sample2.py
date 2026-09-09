import argparse
import os
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from safetensors.torch import load_file

from template import template_dict


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_MAX_SEQUENCE_LENGTH = 512


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_flux_pipeline(model_id, device, torch_dtype):
    pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch_dtype).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    try:
        pipe.set_progress_bar_config(disable=False)
    except AttributeError:
        pass
    return pipe


def prepare_shared_latents(pipe, seeds, args):
    channels = pipe.transformer.config.in_channels // 4
    latents = {}
    for seed in seeds:
        prepared = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=channels,
            height=args.height,
            width=args.width,
            dtype=pipe.transformer.dtype,
            device=pipe.device,
            generator=torch.Generator(device=pipe.device).manual_seed(seed),
            latents=None,
        )
        latents[seed] = (prepared[0] if isinstance(prepared, tuple) else prepared).clone()
    return latents


def flux_generate(pipe, prompts, seeds, args, desc=None, latents_by_seed=None, stream=None):
    images = []
    stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_context:
        for prompt, seed in zip(prompts, seeds):
            kwargs = dict(
                prompt=prompt,
                num_inference_steps=args.total_timesteps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                max_sequence_length=args.max_sequence_length,
            )
            if latents_by_seed is None:
                kwargs["generator"] = torch.Generator(device=pipe.device).manual_seed(int(seed))
            else:
                kwargs["latents"] = latents_by_seed[int(seed)].clone()
            images.append(pipe(**kwargs).images[0])
    if stream is not None:
        stream.synchronize()
    if desc is not None:
        print(f"{desc}: generated {len(images)} images")
    return images


def load_edit_weights(pipe_edit, edit_ckpt):
    edit_state_dict = load_file(edit_ckpt, device="cpu")
    transformer_state = pipe_edit.transformer.state_dict()
    print(f"Loading edited transformer weights from {edit_ckpt}")
    print(f"Edited checkpoint keys: {len(edit_state_dict)}")
    max_pre_load_diff = 0.0
    max_post_load_diff = 0.0
    for key, value in edit_state_dict.items():
        state_key = key[len("transformer."):] if key.startswith("transformer.") else key
        if state_key not in transformer_state:
            raise KeyError(f"Edited weight '{key}' is not in the FLUX transformer state dict")
        expected = transformer_state[state_key]
        loaded_value = value.to(device=expected.device, dtype=expected.dtype)
        pre_load_diff = (expected.float() - loaded_value.float()).norm()
        expected.copy_(loaded_value)
        post_load_diff = (expected.float() - loaded_value.float()).norm()
        max_pre_load_diff = max(max_pre_load_diff, pre_load_diff.item())
        max_post_load_diff = max(max_post_load_diff, post_load_diff.item())
    print(
        f"Loaded {len(edit_state_dict)} edited FLUX transformer weights | "
        f"max_pre_load_diff={max_pre_load_diff:.6f} | "
        f"max_post_load_diff={max_post_load_diff:.6f}"
    )


def combine_images_horizontally(images):
    widths, heights = zip(*(img.size for img in images))
    new_img = Image.new("RGB", (sum(widths), max(heights)))
    for i, img in enumerate(images):
        new_img.paste(img, (sum(widths[:i]), 0))
    return new_img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="")
    parser.add_argument("--sd_ckpt", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--mode", type=str, default="original", help="original, edit")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--total_timesteps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--erase_type", type=str, default="")
    parser.add_argument("--target_concept", type=str, default="")
    parser.add_argument("--contents", type=str, default="")
    parser.add_argument("--edit_ckpt", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--nudity_path", type=str, default=None)
    parser.add_argument("--coco_path", type=str, default=None)
    parser.add_argument("--max_num", type=int, default=None)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num_samples must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.total_timesteps < 1:
        parser.error("--total_timesteps must be at least 1")

    diffusers_logging.set_verbosity_error()
    diffusers_logging.enable_progress_bar()

    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    seed_everything(args.seed, True)

    model_id = args.model_id or args.sd_ckpt
    contents = [x.strip() for x in args.contents.split(",") if x.strip()]
    if "edit" in mode_list:
        sampled_contents = []
        for content in contents:
            check_path = os.path.join(
                args.save_root,
                args.target_concept.replace(", ", "_"),
                content,
                "edit",
            )
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != expected_count_for_content(content, args):
                sampled_contents.append(content)
        contents = sampled_contents
        if not contents:
            return

    paired_mode = "original" in mode_list and "edit" in mode_list
    if paired_mode:
        if not args.device.startswith("cuda"):
            raise ValueError("Paired parallel generation requires a CUDA --device")
        if args.edit_ckpt is None:
            raise ValueError("--edit_ckpt is required when --mode includes edit")
        pipe_original = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])
        pipe_edit = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])
        load_edit_weights(pipe_edit, args.edit_ckpt)
        original_stream = torch.cuda.Stream(device=args.device)
        edit_stream = torch.cuda.Stream(device=args.device)
    elif "original" in mode_list:
        pipe_original = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])
    elif "edit" in mode_list:
        if args.edit_ckpt is None:
            raise ValueError("--edit_ckpt is required when --mode includes edit")
        pipe_edit = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])
        load_edit_weights(pipe_edit, args.edit_ckpt)

    def generate_pass(pass_mode, pipe):
        for content in contents:
            dataset = AdaDataset(content=content, args=args)
            dataloader = DataLoader(dataset, batch_size=bs, drop_last=False)
            save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), content)
            os.makedirs(os.path.join(save_path, pass_mode), exist_ok=True)
            for count, data in enumerate(tqdm(dataloader, desc=f"{content} {pass_mode} batches")):
                prompts = list(data["prompt"])
                seeds = [int(x) for x in data["seed"]]
                filenames = list(data["filename"])
                images = flux_generate(
                    pipe=pipe,
                    prompts=prompts,
                    seeds=seeds,
                    args=args,
                    desc=f"{count * len(prompts)} x prompts | {pass_mode}",
                )
                for image, save_filename in zip(images, filenames):
                    image.save(os.path.join(save_path, pass_mode, save_filename))

    def generate_paired():
        with ThreadPoolExecutor(max_workers=2) as executor:
            for content in contents:
                dataset = AdaDataset(content=content, args=args)
                dataloader = DataLoader(dataset, batch_size=bs, drop_last=False)
                save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), content)
                for pass_mode in ("original", "edit"):
                    os.makedirs(os.path.join(save_path, pass_mode), exist_ok=True)
                for count, data in enumerate(tqdm(dataloader, desc=f"{content} paired batches")):
                    prompts = list(data["prompt"])
                    seeds = [int(x) for x in data["seed"]]
                    filenames = list(data["filename"])
                    shared_latents = prepare_shared_latents(pipe_original, seeds, args)
                    original_future = executor.submit(
                        flux_generate, pipe_original, prompts, seeds, args,
                        f"{count * len(prompts)} x prompts | original", shared_latents, original_stream,
                    )
                    edit_future = executor.submit(
                        flux_generate, pipe_edit, prompts, seeds, args,
                        f"{count * len(prompts)} x prompts | edit", shared_latents, edit_stream,
                    )
                    for pass_mode, images in (("original", original_future.result()), ("edit", edit_future.result())):
                        for image, save_filename in zip(images, filenames):
                            image.save(os.path.join(save_path, pass_mode, save_filename))

    if paired_mode:
        generate_paired()
    elif "original" in mode_list:
        generate_pass("original", pipe_original)
    elif "edit" in mode_list:
        generate_pass("edit", pipe_edit)

    if "original" in mode_list and "edit" in mode_list:
        for content in contents:
            save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), content)
            original_dir = os.path.join(save_path, "original")
            edit_dir = os.path.join(save_path, "edit")
            combine_dir = os.path.join(save_path, "combine")
            os.makedirs(combine_dir, exist_ok=True)
            for save_filename in os.listdir(original_dir):
                original_path = os.path.join(original_dir, save_filename)
                edit_path = os.path.join(edit_dir, save_filename)
                if not os.path.isfile(edit_path):
                    continue
                with Image.open(original_path) as original_image, Image.open(edit_path) as edit_image:
                    combined = combine_images_horizontally([original_image.convert("RGB"), edit_image.convert("RGB")])
                    combined.save(os.path.join(combine_dir, save_filename.replace(".png", ".jpg")))


def expected_count_for_content(content, args):
    if content in ["nudity", "coco", "erase", "retain"]:
        return len(AdaDataset(content=content, args=args))
    return len(template_dict[args.erase_type]) * args.num_samples


class AdaDataset(Dataset):
    def __init__(self, content, args):
        self.content = content
        self.prompt_list, self.idx, self.seed, self.filename = [], [], [], []

        if content == "nudity":
            data_path = Path(args.nudity_path or os.path.join(args.data_root, "NSFW.csv"))
            for row_index, raw_line in enumerate(data_path.read_text().splitlines()):
                prompt = raw_line.strip().rstrip(",").strip()
                if prompt.startswith('"') and prompt.endswith('"'):
                    prompt = prompt[1:-1]
                if not prompt:
                    continue
                self.prompt_list.append(prompt)
                self.idx.append(row_index)
                self.seed.append(args.seed + row_index)
                self.filename.append(f"{row_index}_{self._safe_name(prompt, 100)}.png")
            if args.max_num is not None:
                self.prompt_list = self.prompt_list[:args.max_num]
                self.idx = self.idx[:args.max_num]
                self.seed = self.seed[:args.max_num]
                self.filename = self.filename[:args.max_num]

        elif content == "coco":
            data_path = args.coco_path or os.path.join(args.data_root, "mscoco.csv")
            data = pd.read_csv(data_path)
            data = data.iloc[:1000 if args.max_num is None else args.max_num]
            self.prompt_list = list(data["text"])
            self.idx = [int(x) for x in data["image_id"]]
            self.seed = [args.seed] * len(self.prompt_list)
            self.filename = [f"COCO_val2014_{int(x):012}.png" for x in self.idx]

        elif content in ["erase", "retain"]:
            data_path = args.dataset_path or os.path.join(args.data_root, f"{args.erase_type}.csv")
            data = pd.read_csv(data_path)
            if "type" not in data.columns:
                raise ValueError(f"{data_path} must contain a 'type' column for '{content}' sampling")
            data = data[data["type"] == content]
            if args.max_num is not None:
                data = data.iloc[:args.max_num]
            self.prompt_list = list(data["text"] if "text" in data.columns else data["prompt"])
            self.idx = list(data["id"] if "id" in data.columns else range(len(self.prompt_list)))
            self.seed = [int(x) for x in (data["seed"] if "seed" in data.columns else [args.seed] * len(self.prompt_list))]
            self.filename = [
                f"{self._safe_name(prompt)}_{idx}.png"
                for prompt, idx in zip(self.prompt_list, self.idx)
            ]

        else:
            if args.prompts is None:
                prompt_templates = template_dict[args.erase_type]
            else:
                prompt_templates = [x.strip() for x in args.prompts.split(";") if x.strip()]
            prompts = [template.format(content) for template in prompt_templates]
            for prompt in prompts:
                for sample_idx in range(args.num_samples):
                    self.prompt_list.append(prompt)
                    self.idx.append(sample_idx)
                    self.seed.append(args.seed + sample_idx)
                    self.filename.append(f"{self._safe_name(prompt)}_{sample_idx}.png")

    def _safe_name(self, prompt, max_len=140):
        filename = re.sub(r"[^\w\s]", "", str(prompt), flags=re.UNICODE).replace(" ", "_")
        filename = re.sub(r"_+", "_", filename).strip("_")
        return (filename or "prompt")[:max_len]

    def __getitem__(self, idx):
        return {
            "prompt": self.prompt_list[idx],
            "idx": self.idx[idx],
            "seed": self.seed[idx],
            "filename": self.filename[idx],
        }

    def __len__(self):
        return len(self.prompt_list)


if __name__ == "__main__":
    main()
