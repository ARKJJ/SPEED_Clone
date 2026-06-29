import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
from PIL import Image

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file

try:
    from src.template import template_dict
except ModuleNotFoundError:
    speed_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SPEED-main"))
    if speed_src not in sys.path:
        sys.path.append(speed_src)
    from src.template import template_dict

try:
    from src.utils import seed_everything
except ModuleNotFoundError:
    def seed_everything(seed, deterministic=False):
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def expected_output_count(prompt_templates, num_samples):
    return len(prompt_templates) * num_samples


def load_edit_state_dict(edit_path, device="cpu"):
    if edit_path.endswith(".safetensors"):
        return load_file(edit_path, device=device)
    return torch.load(edit_path, map_location=device)


def apply_transformer_edit_weights(pipe, edit_state_dict, strict=True):
    transformer_state = pipe.transformer.state_dict()
    loaded_keys = []

    for key, value in edit_state_dict.items():
        transformer_key = key[len("transformer."):] if key.startswith("transformer.") else key
        if transformer_key not in transformer_state:
            message = f"Edit weight key '{key}' not found in FLUX transformer state_dict"
            if strict:
                raise KeyError(message)
            print(f"Warning: {message}; skipped.")
            continue
        expected = transformer_state[transformer_key]
        if expected.shape != value.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': checkpoint {tuple(value.shape)} vs model {tuple(expected.shape)}"
            )
        expected.copy_(value.to(device=expected.device, dtype=expected.dtype))
        loaded_keys.append(transformer_key)

    if not loaded_keys:
        raise RuntimeError("No edit weights were loaded into the FLUX transformer")
    print(f"Loaded {len(loaded_keys)} edited FLUX transformer weights.")
    return loaded_keys


def load_flux_pipeline(model_id, device, torch_dtype):
    pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch_dtype).to(device)
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    return pipe


def flux_generate(pipe, prompt, seeds, args, desc=None):
    images = []
    for seed in seeds:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt,
            generator=generator,
            num_inference_steps=args.total_timesteps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
        )
        images.append(result.images[0])
    if desc is not None:
        print(f"{desc}: generated {len(images)} images")
    return images


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, edit')
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--total_timesteps', type=int, default=4, help='The total timesteps of the sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate' )
    parser.add_argument('--batch_size', type=int, default=10, help='Kept for SPEED CLI compatibility')
    parser.add_argument('--prompts', type=str, default=None)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--max_sequence_length', type=int, default=None)
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, default=None)
    parser.add_argument('--strict_edit_load', action='store_true', default=False)
    args = parser.parse_args()

    mode_list = args.mode.replace(' ', '').split(',')
    if not set(mode_list).issubset({'original', 'edit'}):
        raise ValueError("--mode must contain only 'original' and/or 'edit'")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    model_id = args.model_id or args.sd_ckpt
    if args.max_sequence_length is None:
        args.max_sequence_length = 256 if 'schnell' in model_id.lower() else 512
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    # region [If certain concept is already sampled, then skip it.]
    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',') if item.strip()]
    if 'edit' in mode_list:
        prompt_templates = template_dict[args.erase_type] if args.prompts is None else args.prompts.split(';')
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'edit')
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != expected_output_count(prompt_templates, args.num_samples):
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0: sys.exit()
    # endregion

    # region [Prepare Models]
    pipe = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])
    pipe_edit = None
    if 'edit' in mode_list:#只替换transformer中CE_Flux更新过的权重
        pipe_edit = copy.deepcopy(pipe) if 'original' in mode_list else pipe
        edit_path = args.edit_ckpt or os.path.join("models", sorted(os.listdir("models"))[-1])
        edit_state_dict = load_edit_state_dict(edit_path, device='cpu')
        apply_transformer_edit_weights(pipe_edit, edit_state_dict, strict=args.strict_edit_load)
    # endregion

    # Sampling process
    seed_everything(args.seed, True)
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]
    bs = args.batch_size
    for i in range((args.num_samples + bs - 1) // bs):#采样循环
        start_idx = i * bs
        end_idx = min(start_idx + bs, args.num_samples)
        seeds = [args.seed + sample_idx for sample_idx in range(start_idx, end_idx)]
        for concept, prompts in zip(concept_list, prompt_list):
            for count, prompt in enumerate(prompts):

                save_images = {}

                if 'original' in mode_list:
                    save_images['original'] = flux_generate(pipe=pipe,
                                                   prompt=prompt,
                                                   seeds=seeds,
                                                   args=args,
                                                   desc=f"{count} x {prompt} | original")
                if 'edit' in mode_list:
                    save_images['edit'] = flux_generate(pipe=pipe_edit,
                                               prompt=prompt,
                                               seeds=seeds,
                                               args=args,
                                               desc=f"{count} x {prompt} | edit")
                                        
                save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)
                for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1: os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                # Save images
                def combine_images_horizontally(Images):
                    widths, heights = zip(*(img.size for img in Images))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for i, img in enumerate(Images): new_img.paste(img, (sum(widths[:i]), 0))
                    return new_img
                for idx in range(len(save_images[mode_list[0]])):
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{int(idx + start_idx)}.png"
                    images_to_combine = []
                    for mode in mode_list: 
                        save_images[mode][idx].save(os.path.join(save_path, mode, save_filename))
                        images_to_combine.append(save_images[mode][idx])
                    if len(mode_list) > 1:
                        img_combined = combine_images_horizontally(images_to_combine)
                        img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))


if __name__ == '__main__':
    main()
