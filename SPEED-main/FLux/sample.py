import warnings  
warnings.filterwarnings("ignore")  
import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  
import re  
import copy  
import argparse 
from PIL import Image  

import random
import torch
import numpy as np
from diffusers import DiffusionPipeline  
from safetensors.torch import load_file  
from template import template_dict  

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
    return pipe 


def flux_generate(pipe, prompt, seeds, args, desc=None):
    images = []
    for seed in seeds:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
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
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--mode', type=str, default='original', help='original, edit')
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--total_timesteps', type=int, default=20, help='The total timesteps of the sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate' )
    parser.add_argument('--batch_size', type=int, default=10, help='Kept for SPEED CLI compatibility')
    parser.add_argument('--prompts', type=str, default=None)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--max_sequence_length', type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, default=None)
    args = parser.parse_args()

    mode_list = args.mode.replace(' ', '').split(',')
    model_id = args.model_id or args.sd_ckpt
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    concept_list = [item.strip() for item in args.contents.split(',') if item.strip()]
    if len(concept_list) == 0:
        return

    pipe = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])
    pipe_edit = None
    if 'edit' in mode_list:
        pipe_edit = copy.deepcopy(pipe) if 'original' in mode_list else pipe
        if args.edit_ckpt is None:
            raise ValueError("--edit_ckpt is required when --mode includes edit")
        edit_path = args.edit_ckpt
        edit_state_dict = load_file(edit_path, device='cpu')
        print(f"Loading edited transformer weights from {edit_path}")
        print(f"Edited checkpoint keys: {len(edit_state_dict)}")
        transformer_state = pipe_edit.transformer.state_dict()
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
            f"Loaded {len(edit_state_dict)} edited transformer weights | "
            f"max_pre_load_diff={max_pre_load_diff:.6f} | "
            f"max_post_load_diff={max_post_load_diff:.6f}"
        )

    seed_everything(args.seed, True)
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]
    bs = args.batch_size
    for i in range((args.num_samples + bs - 1) // bs):
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
                for mode in mode_list:
                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1:
                    os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                def combine_images_horizontally(Images):
                    widths, heights = zip(*(img.size for img in Images))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for i, img in enumerate(Images):
                        new_img.paste(img, (sum(widths[:i]), 0))
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
