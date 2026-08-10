import argparse
import os
import random
import re
import warnings

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from diffusers import DiffusionPipeline


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_filename(prompt, max_len=120):
    filename = re.sub(r"[^\w\s-]", "", str(prompt), flags=re.UNICODE).replace(" ", "_")
    filename = re.sub(r"_+", "_", filename).strip("_")
    return (filename or "flux_image")[:max_len]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate one image from one prompt with FLUX.2-klein-4B.")
    parser.add_argument("prompt", type=str, help="Text prompt used for image generation.")
    parser.add_argument("--output", type=str, default=None, help="Output image path. Defaults to outputs/<prompt>_<seed>.png")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    seed_everything(args.seed, deterministic=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        safety_checker=None,
        torch_dtype=dtype_map[args.torch_dtype],
    ).to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        max_sequence_length=args.max_sequence_length,
    ).images[0]

    output = args.output
    if output is None:
        output = os.path.join("FLux", "outputs", f"{safe_filename(args.prompt)}_{args.seed}.png")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    image.save(output)
    print(f"Saved image to: {output}")


if __name__ == "__main__":
    main()
