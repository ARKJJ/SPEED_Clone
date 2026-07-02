import warnings  # 导入 warnings，用于控制警告输出
warnings.filterwarnings("ignore")  # 忽略运行时警告，减少采样日志噪声
import os, sys, pdb  # 导入系统路径、退出接口和调试器；pdb 当前未显式使用
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置 Hugging Face 镜像站，便于国内下载模型
import re  # 导入正则表达式库，用于清理保存文件名
import copy  # 导入 copy，用于复制 pipeline 以同时保留 original 和 edit 模型
import argparse  # 导入命令行参数解析库
from PIL import Image  # 导入 PIL Image，用于拼接和保存图片

import torch  # 导入 PyTorch，用于随机种子、dtype 和 generator
from diffusers import DiffusionPipeline  # 导入 diffusers 的通用扩散模型 pipeline
from safetensors.torch import load_file  # 导入 safetensors 读取函数，用于加载编辑权重
from src.template import template_dict  # 导入 SPEED 的 prompt 模板字典
from src.utils import seed_everything  # 导入随机种子设置函数


def expected_output_count(prompt_templates, num_samples):  # 计算某个 concept 理论上应该生成多少张图
    return len(prompt_templates) * num_samples  # prompt 模板数量乘每个模板采样数量


def load_edit_state_dict(edit_path, device="cpu"):  # 加载 CE_Flux 保存的编辑权重字典
    if edit_path.endswith(".safetensors"):  # 如果权重文件是 safetensors 格式
        return load_file(edit_path, device=device)  # 使用 safetensors 的 load_file 加载到指定设备
    return torch.load(edit_path, map_location=device)  # 否则按 PyTorch checkpoint 方式加载


def apply_transformer_edit_weights(pipe, edit_state_dict, strict=True):  # 将编辑权重加载进 FLUX pipeline 的 transformer
    transformer_state = pipe.transformer.state_dict()  # 取出当前 transformer 的全部权重字典
    loaded_keys = []  # 记录成功加载的权重 key

    for key, value in edit_state_dict.items():  # 遍历编辑 checkpoint 中的每个权重
        transformer_key = key[len("transformer."):] if key.startswith("transformer.") else key  # 兼容带 transformer. 前缀和不带前缀的 key
        if transformer_key not in transformer_state:  # 如果该 key 不存在于当前 FLUX transformer
            message = f"Edit weight key '{key}' not found in FLUX transformer state_dict"  # 构造错误/警告信息
            if strict:  # 如果严格加载模式开启
                raise KeyError(message)  # 直接报错，避免漏加载权重
            print(f"Warning: {message}; skipped.")  # 非严格模式下打印警告并跳过
            continue  # 跳过当前无法匹配的权重
        expected = transformer_state[transformer_key]  # 取出模型中对应权重 tensor
        if expected.shape != value.shape:  # 检查 checkpoint 权重形状是否和模型权重一致
            raise ValueError(  # 形状不一致说明 checkpoint 与模型结构不匹配
                f"Shape mismatch for '{key}': checkpoint {tuple(value.shape)} vs model {tuple(expected.shape)}"  # 报告 checkpoint 和模型的形状
            )  # ValueError 结束
        expected.copy_(value.to(device=expected.device, dtype=expected.dtype))  # 将编辑权重转到模型权重设备和 dtype 后原地复制
        loaded_keys.append(transformer_key)  # 记录成功加载的 transformer key

    if not loaded_keys:  # 如果没有任何权重成功加载
        raise RuntimeError("No edit weights were loaded into the FLUX transformer")  # 直接报错，避免误以为 edit 生效
    print(f"Loaded {len(loaded_keys)} edited FLUX transformer weights.")  # 打印成功加载的编辑权重数量
    return loaded_keys  # 返回成功加载的 key 列表，便于调试检查


def load_flux_pipeline(model_id, device, torch_dtype):  # 加载 FLUX pipeline 并做基础显存优化
    pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch_dtype).to(device)  # 从模型名/路径加载 pipeline 并移到设备
    if hasattr(pipe, "vae"):  # 如果 pipeline 中存在 VAE 模块
        pipe.vae.enable_slicing()  # 开启 VAE slicing，降低解码显存占用
        pipe.vae.enable_tiling()  # 开启 VAE tiling，适合较大分辨率时省显存
    return pipe  # 返回加载好的 FLUX pipeline


def flux_generate(pipe, prompt, seeds, args, desc=None):  # 使用指定 pipeline、prompt 和多个 seed 生成图片
    images = []  # 保存生成出的 PIL 图片
    for seed in seeds:  # 遍历本批次要使用的随机种子
        generator = torch.Generator(device=pipe.device).manual_seed(seed)  # 为当前 seed 创建随机数生成器
        result = pipe(  # 调用 FLUX pipeline 生成图片
            prompt,  # 当前文本 prompt
            generator=generator,  # 当前 seed 对应的随机生成器
            num_inference_steps=args.total_timesteps,  # 采样去噪步数
            guidance_scale=args.guidance_scale,  # guidance scale，schnell 常用 0.0
            height=args.height,  # 输出图片高度
            width=args.width,  # 输出图片宽度
            max_sequence_length=args.max_sequence_length,  # 文本最大 token 长度
        )  # pipeline 生成结束
        images.append(result.images[0])  # 取 batch 中第一张图加入结果列表
    if desc is not None:  # 如果提供了描述信息
        print(f"{desc}: generated {len(images)} images")  # 打印当前 prompt 生成数量
    return images  # 返回当前 prompt 对应的图片列表


@torch.no_grad()  # 关闭整个 main 采样过程的梯度计算
def main():  # 命令行采样入口函数

    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Base Config  # 分段：基础路径、模型和设备配置
    parser.add_argument('--save_root', type=str, default='')  # 保存生成结果的根目录
    parser.add_argument('--sd_ckpt', type=str, default="black-forest-labs/FLUX.1-schnell")  # 默认 FLUX 模型名
    parser.add_argument('--model_id', type=str, default=None)  # 可选模型名；如果提供则覆盖 sd_ckpt
    parser.add_argument('--seed', type=int, default=0)  # 起始随机种子
    parser.add_argument('--device', type=str, default='cuda:0')  # 运行设备
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])  # 模型加载 dtype
    # Sampling Config  # 分段：采样行为配置
    parser.add_argument('--mode', type=str, default='original', help='original, edit')  # 采样模式，可生成原模型、编辑模型或两者
    parser.add_argument('--guidance_scale', type=float, default=0.0)  # guidance scale 参数
    parser.add_argument('--total_timesteps', type=int, default=4, help='The total timesteps of the sampling process')  # 采样去噪步数
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate' )  # 每个 prompt 生成几张图
    parser.add_argument('--batch_size', type=int, default=10, help='Kept for SPEED CLI compatibility')  # 每轮处理多少 seed，主要保持 SPEED CLI 兼容
    parser.add_argument('--prompts', type=str, default=None)  # 用户自定义 prompt 模板，用分号分隔
    parser.add_argument('--height', type=int, default=512)  # 生成图片高度
    parser.add_argument('--width', type=int, default=512)  # 生成图片宽度
    parser.add_argument('--max_sequence_length', type=int, default=None)  # 文本最大 token 长度；默认按模型类型自动设置
    # Erasing Config  # 分段：概念与编辑 checkpoint 配置
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')  # 擦除类型，用于从 template_dict 选择模板
    parser.add_argument('--target_concept', type=str, default='')  # 目标概念名，用于保存路径命名
    parser.add_argument('--contents', type=str, default='')  # 实际要采样的 concept 列表，逗号分隔
    parser.add_argument('--edit_ckpt', type=str, default=None)  # 编辑权重 checkpoint 路径
    parser.add_argument('--strict_edit_load', action='store_true', default=False)  # 是否严格检查编辑权重 key 必须全部匹配
    args = parser.parse_args()  # 解析命令行参数

    mode_list = args.mode.replace(' ', '').split(',')  # 去掉空格并按逗号拆分采样模式
    if not set(mode_list).issubset({'original', 'edit'}):  # 检查 mode 是否只包含 original/edit
        raise ValueError("--mode must contain only 'original' and/or 'edit'")  # 非法 mode 直接报错
    if args.num_samples <= 0:  # 检查采样数量是否合法
        raise ValueError("--num_samples must be positive")  # num_samples 必须大于 0
    model_id = args.model_id or args.sd_ckpt  # 优先使用 model_id，否则使用 sd_ckpt
    if args.max_sequence_length is None:  # 如果用户没有显式指定文本最大长度
        args.max_sequence_length = 256 if 'schnell' in model_id.lower() else 512  # schnell 用 256，其它 FLUX 模型用 512
    dtype_map = {  # 字符串 dtype 到 torch dtype 的映射
        'float16': torch.float16,  # float16 半精度
        'bfloat16': torch.bfloat16,  # bfloat16 半精度
        'float32': torch.float32,  # float32 单精度
    }  # dtype 映射结束

    # region [If certain concept is already sampled, then skip it.]  # 分段：检查已生成结果，避免重复采样
    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',') if item.strip()]  # 将 contents 拆成 concept 列表并去空项
    if 'edit' in mode_list:  # 如果要生成 edit 结果
        prompt_templates = template_dict[args.erase_type] if args.prompts is None else args.prompts.split(';')  # 获取模板，用于计算应有图片数量
        for concept in concept_list_tmp:  # 遍历每个待采样 concept
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'edit')  # 构造该 concept 的 edit 保存目录
            os.makedirs(check_path, exist_ok=True)  # 确保目录存在，便于统计已有文件
            if len(os.listdir(check_path)) != expected_output_count(prompt_templates, args.num_samples):  # 如果已有文件数不等于期望输出数
                concept_list.append(concept)  # 说明该 concept 还未完整采样，需要加入采样列表
    else:  # 如果只生成 original
        concept_list = concept_list_tmp  # 不检查 edit 目录，直接采样所有 concept
    if len(concept_list) == 0: sys.exit()  # 如果没有需要采样的 concept，直接退出
    # endregion  # 跳过已完成 concept 的分段结束

    # region [Prepare Models]  # 分段：加载原模型和编辑模型
    pipe = load_flux_pipeline(model_id, args.device, dtype_map[args.torch_dtype])  # 加载原始 FLUX pipeline
    pipe_edit = None  # 初始化编辑模型 pipeline
    if 'edit' in mode_list:  # 如果需要生成编辑后图片
        pipe_edit = copy.deepcopy(pipe) if 'original' in mode_list else pipe  # 如果同时要 original/edit，就复制一份；否则直接复用 pipe
        edit_path = args.edit_ckpt or os.path.join("models", sorted(os.listdir("models"))[-1])  # 使用用户指定 checkpoint，或默认 models 下最新文件
        edit_state_dict = load_edit_state_dict(edit_path, device='cpu')  # 将编辑权重先加载到 CPU
        apply_transformer_edit_weights(pipe_edit, edit_state_dict, strict=args.strict_edit_load)  # 将编辑权重写入编辑 pipeline 的 transformer
    # endregion  # 模型准备分段结束

    # Sampling process  # 分段：构造 prompts 并循环生成图片
    seed_everything(args.seed, True)  # 设置全局随机种子，提升结果可复现性
    if args.prompts is None:  # 如果用户没有提供自定义 prompts
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]  # 使用 template_dict 中对应 erase_type 的模板
    else:  # 如果用户提供了自定义 prompts
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]  # 按分号拆分模板并填入 concept
    bs = args.batch_size  # 取出 batch_size，用于按 seed 分批生成
    for i in range((args.num_samples + bs - 1) // bs):  # 外层循环按 batch_size 切分 num_samples
        start_idx = i * bs  # 当前批次起始样本编号
        end_idx = min(start_idx + bs, args.num_samples)  # 当前批次结束样本编号，不超过 num_samples
        seeds = [args.seed + sample_idx for sample_idx in range(start_idx, end_idx)]  # 为当前批次构造 seed 列表
        for concept, prompts in zip(concept_list, prompt_list):  # 同时遍历 concept 和其对应 prompt 列表
            for count, prompt in enumerate(prompts):  # 遍历当前 concept 的每个 prompt 模板结果

                save_images = {}  # 保存当前 prompt 下 original/edit 生成的图片列表

                if 'original' in mode_list:  # 如果需要生成原模型图片
                    save_images['original'] = flux_generate(pipe=pipe,  # 使用原始 pipeline
                                                   prompt=prompt,  # 当前 prompt
                                                   seeds=seeds,  # 当前批次 seeds
                                                   args=args,  # 采样参数
                                                   desc=f"{count} x {prompt} | original")  # 日志描述
                if 'edit' in mode_list:  # 如果需要生成编辑模型图片
                    save_images['edit'] = flux_generate(pipe=pipe_edit,  # 使用编辑后的 pipeline
                                               prompt=prompt,  # 当前 prompt
                                               seeds=seeds,  # 当前批次 seeds
                                               args=args,  # 采样参数
                                               desc=f"{count} x {prompt} | edit")  # 日志描述

                save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)  # 构造当前 concept 的保存目录
                for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)  # 为 original/edit 分别创建保存目录
                if len(mode_list) > 1: os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)  # 如果同时生成两种模式，就创建横向拼接图目录

                # Save images  # 分段：保存单张图和 original/edit 拼接图
                def combine_images_horizontally(Images):  # 将多张 PIL 图片横向拼接成一张图
                    widths, heights = zip(*(img.size for img in Images))  # 取出每张图的宽高
                    new_img = Image.new('RGB', (sum(widths), max(heights)))  # 创建总宽度为宽度和、高度为最大高度的新图
                    for i, img in enumerate(Images): new_img.paste(img, (sum(widths[:i]), 0))  # 将每张图粘贴到对应横向位置
                    return new_img  # 返回拼接后的图片
                for idx in range(len(save_images[mode_list[0]])):  # 遍历当前批次中每个 seed 对应的图片编号
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{int(idx + start_idx)}.png"  # 根据 prompt 和样本编号生成安全文件名
                    images_to_combine = []  # 保存要横向拼接的 original/edit 图片
                    for mode in mode_list:  # 遍历 original/edit 模式
                        save_images[mode][idx].save(os.path.join(save_path, mode, save_filename))  # 保存当前模式下的单张图片
                        images_to_combine.append(save_images[mode][idx])  # 将图片加入待拼接列表
                    if len(mode_list) > 1:  # 如果同时生成 original 和 edit
                        img_combined = combine_images_horizontally(images_to_combine)  # 横向拼接 original/edit 图片
                        img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))  # 将拼接图保存为 jpg


if __name__ == '__main__':  # 只有直接运行该脚本时才执行 main
    main()  # 调用命令行主函数
