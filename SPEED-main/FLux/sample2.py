import os, re, copy, argparse, random, warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline
from safetensors.torch import load_file

from template import template_dict


DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flux_generate(pipe, prompts, seeds, args, desc=None):
    images = []
    for prompt, seed in zip(prompts, seeds):
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
        image = pipe(
            prompt,
            generator=generator,
            num_inference_steps=args.total_timesteps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
        ).images[0]
        images.append(image)
    if desc is not None:
        print(f"{desc}: generated {len(images)} images")
    return images


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument("--save_root", type=str, default="")
    parser.add_argument("--sd_ckpt", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    # Sampling Config
    parser.add_argument("--mode", type=str, default="original", help="original, edit")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--total_timesteps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--prompts", type=str, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    # Erasing Config
    parser.add_argument("--erase_type", type=str, default="")
    parser.add_argument("--target_concept", type=str, default="")
    parser.add_argument("--contents", type=str, default="")
    parser.add_argument("--edit_ckpt", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--i2p_path", type=str, default=None)
    parser.add_argument("--coco_path", type=str, default=None)
    parser.add_argument("--max_num", type=int, default=None)
    args = parser.parse_args()

    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    seed_everything(args.seed, True)

    # region [Prepare Models]
    model_id = args.model_id or args.sd_ckpt
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=dtype_map[args.torch_dtype],
    ).to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if "edit" in mode_list:
        pipe_edit = copy.deepcopy(pipe) if "original" in mode_list else pipe
        if args.edit_ckpt is None:
            raise ValueError("--edit_ckpt is required when --mode includes edit")
        edit_state_dict = load_file(args.edit_ckpt, device="cpu")
        transformer_state = pipe_edit.transformer.state_dict()
        for key, value in edit_state_dict.items():
            state_key = key[len("transformer."):] if key.startswith("transformer.") else key
            if state_key not in transformer_state:
                raise KeyError(f"Edited weight '{key}' is not in the FLUX transformer state dict")
            expected = transformer_state[state_key]
            expected.copy_(value.to(device=expected.device, dtype=expected.dtype))
        print(f"Loaded {len(edit_state_dict)} edited FLUX transformer weights.")
    else:
        pipe_edit = None
    # endregion

    def combine_images_horizontally(images):
        widths, heights = zip(*(img.size for img in images))
        new_img = Image.new("RGB", (sum(widths), max(heights)))
        for i, img in enumerate(images):
            new_img.paste(img, (sum(widths[:i]), 0))
        return new_img

    for content in [x.strip() for x in args.contents.split(",") if x.strip()]:
        dataset = AdaDataset(content=content, args=args)
        dataloader = DataLoader(dataset, batch_size=bs, drop_last=False)

        for count, data in enumerate(tqdm(dataloader, desc=f"{content} batches")):
            prompts = list(data["prompt"])
            seeds = [int(x) for x in data["seed"]]
            filenames = list(data["filename"])
            save_images = {}

            if "original" in mode_list:
                save_images["original"] = flux_generate(
                    pipe=pipe,
                    prompts=prompts,
                    seeds=seeds,
                    args=args,
                    desc=f"{count * len(prompts)} x prompts | original",
                )
            if "edit" in mode_list:
                save_images["edit"] = flux_generate(
                    pipe=pipe_edit,
                    prompts=prompts,
                    seeds=seeds,
                    args=args,
                    desc=f"{count * len(prompts)} x prompts | edit",
                )

            save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), content)
            for mode in mode_list:
                os.makedirs(os.path.join(save_path, mode), exist_ok=True)
            if len(mode_list) > 1:
                os.makedirs(os.path.join(save_path, "combine"), exist_ok=True)

            for idx, save_filename in enumerate(filenames):
                images_to_combine = []
                for mode in mode_list:
                    save_images[mode][idx].save(os.path.join(save_path, mode, save_filename))
                    images_to_combine.append(save_images[mode][idx])
                if len(mode_list) > 1:
                    img_combined = combine_images_horizontally(images_to_combine)
                    img_combined.save(os.path.join(save_path, "combine", save_filename.replace(".png", ".jpg")))


class AdaDataset(Dataset):
    def __init__(self, content, args):
        self.content = content
        self.prompt_list, self.idx, self.seed, self.filename = [], [], [], []

        if content == "nudity":
            data_path = args.i2p_path or os.path.join(args.data_root, "i2p_benchmark.csv")
            data = pd.read_csv(data_path)
            if args.max_num is not None:
                data = data.iloc[:args.max_num]
            self.prompt_list = list(data["prompt"])
            self.idx = list(range(len(self.prompt_list)))
            self.seed = [int(x) for x in data["sd_seed"]]
            self.filename = [
                f"{i}_{self._safe_name(prompt, 100)}.png"
                for i, prompt in zip(self.idx, self.prompt_list)
            ]

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
