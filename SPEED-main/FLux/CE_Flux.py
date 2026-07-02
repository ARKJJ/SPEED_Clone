import os, re  # 导入操作系统接口 os 和正则表达式库 re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置 Hugging Face 下载镜像地址，便于国内环境拉取模型
import time  # 导入时间库，用于生成默认保存文件名
import torch  # 导入 PyTorch，用于张量计算、模型权重和线性代数
import argparse  # 导入命令行参数解析库
import pandas as pd  # 导入 pandas，用于读取 retain CSV 文件
from diffusers import DiffusionPipeline  # 导入 diffusers 的通用扩散模型 pipeline 类
from safetensors.torch import save_file  # 导入 safetensors 保存函数，用于保存编辑后的权重


ATTENTION_SUFFIXES = {  # 定义 Q/K/V 参数名到 FLUX attention 模块后缀的映射
    "Q": ".attn.add_q_proj",  # Q 表示 query projection，对应 add_q_proj 模块
    "K": ".attn.add_k_proj",  # K 表示 key projection，对应 add_k_proj 模块
    "V": ".attn.add_v_proj",  # V 表示 value projection，对应 add_v_proj 模块
}  # Q/K/V 后缀映射结束


def get_token_id(prompt, tokenizer=None, max_sequence_length=None, return_ids_only=True):  # 将文本 prompt 编码成 tokenizer 输出或 input_ids
    token_ids = tokenizer(prompt,padding="max_length",max_length=max_sequence_length or tokenizer.model_max_length,truncation=True,return_tensors="pt")  # 调用 tokenizer，补齐/截断到最大长度，并返回 PyTorch tensor
    return token_ids.input_ids if return_ids_only else token_ids  # 根据开关返回 input_ids，或返回包含 attention_mask 等信息的完整对象


def _selected_attention_suffixes(params):  # 根据命令行 params 选择要编辑的 Q/K/V 模块后缀
    params = params.upper()  # 将用户输入转成大写，兼容 kv、qkv 这类小写输入
    return [ATTENTION_SUFFIXES[param] for param in "QKV" if param in params]  # 按 Q/K/V 固定顺序返回被选中的模块后缀


def _select_text_attention_modules(transformer, device, params):  # 从 FLUX transformer 中筛选要编辑的 text-side attention 模块
    selected = []  # 保存筛选出的 (模块名, 模块对象)
    suffixes = _selected_attention_suffixes(params)  # 根据 params 得到要匹配的 Q/K/V 后缀
    for name, module in transformer.named_modules():  # 遍历 transformer 的全部子模块及其名字
        if not hasattr(module, "weight") or module.weight is None:  # 如果模块没有权重，就不是可编辑线性层
            continue  # 跳过没有 weight 的模块
        if not any(suffix in name for suffix in suffixes):  # 如果模块名不包含目标 Q/K/V 后缀
            continue  # 跳过非目标 attention 投影层
        if name.startswith("transformer_blocks."):  # 只保留 transformer_blocks 内的模块
            selected.append((name, module.to(device)))  # 将模块移到指定设备，并连同模块名一起保存
    return selected  # 返回所有待编辑模块


def _trace_prompt(pipeline, prompt, token_indices, module_names, args, device, max_sequence_length):  # 对单个 prompt 跑推理并追踪指定模块输入输出
    module_lookup = dict(pipeline.transformer.named_modules())  # 建立模块名到模块对象的字典，方便按名字查模块
    traces = {name: {"inputs": [], "outputs": []} for name in module_names}  # 为每个待追踪模块准备输入/输出记录列表
    handles = []  # 保存 hook 句柄，后面必须用它们移除 hook

    # 分段：注册 hook，用来在 pipeline 前向传播时记录模块输入和输出
    for name in module_names:  # 遍历每个需要追踪的模块名
        module = module_lookup[name]  # 根据模块名找到真实模块对象

        def pre_hook(_module, inputs, module_name=name):  # 定义前向前 hook，在模块计算前记录输入
            traces[module_name]["inputs"].append(inputs[0][:, token_indices, :].detach().float())  # 取目标 token 位置的输入，断开梯度并转 float32

        def out_hook(_module, _inputs, output, module_name=name):  # 定义前向后 hook，在模块计算后记录输出
            output = output[0] if isinstance(output, tuple) else output  # 如果模块输出是 tuple，就取第一个 tensor
            traces[module_name]["outputs"].append(output[:, token_indices, :].detach().float())  # 取目标 token 位置的输出，断开梯度并转 float32

        handles.extend([module.register_forward_pre_hook(pre_hook), module.register_forward_hook(out_hook)])  # 注册输入 hook 和输出 hook，并保存句柄

    # 分段：运行一次 diffusion pipeline，让 hook 自动收集中间激活
    try:  # 用 try/finally 保证即使推理报错也会清理 hook
        generator = torch.Generator(device=device).manual_seed(args.trace_seed)  # 创建指定设备上的随机数生成器，并设置 trace 随机种子
        with torch.no_grad():  # 关闭梯度计算，只做推理和激活追踪
            pipeline(  # 调用 FLUX pipeline 执行一次 latent 输出推理
                prompt,  # 当前要追踪的文本 prompt
                generator=generator,  # 使用固定随机种子生成初始噪声
                num_inference_steps=args.trace_num_steps,  # 设置 trace 时的去噪步数
                guidance_scale=0.0,  # 关闭 classifier-free guidance，减少额外条件干扰
                height=args.trace_resolution,  # 设置 trace 图像高度
                width=args.trace_resolution,  # 设置 trace 图像宽度
                max_sequence_length=max_sequence_length,  # 设置文本最大 token 长度
                output_type="latent",  # 只输出 latent，不解码成图片以节省开销
            )  # pipeline 调用结束，hook 已在内部自动记录输入输出
    finally:  # 无论推理成功还是失败，都执行 hook 清理
        for handle in handles:  # 遍历所有 hook 句柄
            handle.remove()  # 移除 hook，防止后续 trace 重复记录或污染

    # 分段：把多次 hook 记录整理成后续闭式解需要的二维矩阵
    compact = {}  # 保存整理后的 trace 结果
    for name, record in traces.items():  # 遍历每个模块的原始记录
        if not record["inputs"] or not record["outputs"]:  # 如果输入或输出没有被记录到，说明 trace 失败
            raise RuntimeError(f"No trace was collected for module '{name}'")  # 直接报错，避免静默产生错误编辑
        compact[name] = {  # 为当前模块保存整理后的输入输出矩阵
            "inputs": torch.cat(record["inputs"], dim=0).reshape(-1, record["inputs"][0].shape[-1]).T,  # 拼接输入记录并整理成 [hidden_dim, 样本数]
            "outputs": torch.cat(record["outputs"], dim=0).reshape(-1, record["outputs"][0].shape[-1]).T,  # 拼接输出记录并整理成 [hidden_dim, 样本数]
        }  # 当前模块 compact 结果结束
    return compact  # 返回该 prompt 在指定模块上的输入输出 trace


def _closed_form_update(keys, residuals, update_lambda, retain_inputs, retain_threshold=1e-1):  # 用闭式解计算当前模块的权重更新 delta
    retain_inputs = retain_inputs.to(device=keys.device, dtype=keys.dtype)  # 将 retain 输入转到与 keys 相同的设备和数据类型
    covariance = retain_inputs @ retain_inputs.T / retain_inputs.shape[1]  # 计算 retain 输入的协方差/Gram 近似
    U, S, _ = torch.linalg.svd(covariance, full_matrices=False)  # 对 retain 协方差做 SVD，得到方向 U 和奇异值 S
    null_basis = U[:, S < retain_threshold]  # 选择 retain 不敏感的低奇异值方向作为近似零空间
    if null_basis.shape[1] == 0:  # 如果没有找到低奇异值方向
        projector = torch.eye(keys.shape[0], device=keys.device, dtype=keys.dtype)  # 使用单位矩阵，退化为不投影的全空间更新
    else:  # 如果找到了 retain 低影响方向
        projector = null_basis @ null_basis.T  # 构造投影矩阵，将更新限制到 retain 低影响方向
    projected_keys = projector @ keys  # 将 target 输入投影到 retain 低影响方向
    eye = torch.eye(projected_keys.shape[0], device=projected_keys.device, dtype=projected_keys.dtype)  # 创建单位矩阵用于正则化
    system = projected_keys @ projected_keys.T + update_lambda * eye  # 构造岭回归线性系统，lambda 防止解不稳定
    delta = torch.linalg.solve(system.T, (residuals @ projected_keys.T).T).T  # 求解 delta，使 delta @ projected_keys 近似 residuals
    return delta @ projector  # 返回最终权重更新，并确保更新作用在投影子空间中


def _trace_many(pipeline, concepts, token_indices, module_names, args, device, max_sequence_length):  # 批量追踪多个 concept 的指定模块输入输出
    return {  # 返回 concept 到 trace 结果的字典
        concept: _trace_prompt(pipeline, concept, token_indices[concept], module_names, args, device, max_sequence_length)  # 对当前 concept 调用单 prompt trace
        for concept in dict.fromkeys(concepts)  # 对 concepts 去重并保持原顺序
        if token_indices.get(concept)  # 只追踪存在 token 位置且 token 位置非空的 concept
    }  # 批量 trace 字典构造结束


def _mean_outputs(traces, concepts, module_name):  # 计算多个 anchor concept 在指定模块上的平均输出向量
    outputs = [traces[c][module_name]["outputs"] for c in concepts if c in traces and module_name in traces[c]]  # 收集所有可用 concept 的模块 outputs
    return None if not outputs else torch.cat(outputs, dim=1).mean(dim=1, keepdim=True)  # 没有输出则返回 None，否则拼接后沿样本维求平均


def edit_model(args,pipeline,target_concepts,anchor_concepts,retain_texts,device="cuda:0",max_sequence_length=256,):  # 主编辑函数：根据 target/anchor/retain 修改 FLUX attention 权重
    if len(target_concepts) != len(anchor_concepts):  # 检查 target 和 anchor 是否一一对应
        raise ValueError("target_concepts and anchor_concepts must have the same length")  # 数量不一致时直接报错

    # 分段：选择待编辑模块，并预先计算每个模块的 Q/K/V 类型、最终模块和剩余同类模块数
    edit_modules = _select_text_attention_modules(pipeline.transformer, device, args.params)  # 选出要编辑的 Q/K/V text-side attention 模块
    module_names = [name for name, _ in edit_modules]  # 从 (模块名, 模块对象) 中单独取出模块名列表
    selected_suffixes = _selected_attention_suffixes(args.params)  # 得到本次选择的 Q/K/V 后缀列表
    module_suffixes = {  # 建立模块名到其 Q/K/V 后缀类型的映射
        name: next(suffix for suffix in selected_suffixes if suffix in name)  # 找到当前模块名包含的第一个目标后缀
        for name in module_names  # 遍历所有待编辑模块名
    }  # 模块后缀映射结束
    final_modules = {}  # 保存每类 Q/K/V 的最后一个模块名
    remaining_counts = [0] * len(module_names)  # 保存从当前位置开始还剩多少个同类模块
    suffix_counts = {}  # 从后往前统计每类模块已经遇到的数量
    for module_index in range(len(module_names) - 1, -1, -1):  # 从最后一个模块反向遍历到第一个模块
        suffix = module_suffixes[module_names[module_index]]  # 获取当前模块属于 Q/K/V 哪一类
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1  # 当前类型计数加一
        remaining_counts[module_index] = suffix_counts[suffix]  # 记录当前位置之后含当前位置的同类模块数量
        final_modules.setdefault(suffix, module_names[module_index])  # 第一次反向遇到的同类模块就是该类最后模块

    # 分段：清理 retain_texts，避免要保留的文本和要擦除的 target 冲突
    retain_texts = [  # 重新构造过滤后的 retain_texts
        text for text in retain_texts  # 遍历原始 retain 文本
        if not any(re.search(r"\b" + re.escape(concept.lower()) + r"\b", text.lower()) for concept in target_concepts)  # 去掉包含 target 概念完整词的 retain 文本
    ]  # retain 文本过滤结束
    if len(retain_texts) + len(target_concepts) != len(set(retain_texts + target_concepts)):  # 检查 retain 和 target 是否还有完全重复项
        raise ValueError("retain_texts and target_concepts must not overlap")  # 如果仍重叠，直接报错

    # region [Target and Anchor]  # 分段：准备 token 位置，并追踪 anchor 的最终模块输出
    token_indices = {}  # 保存每个 concept 对应要追踪的 token 位置
    for concept in anchor_concepts + retain_texts:  # 先处理 anchor 和 retain 文本
        concept_inputs = get_token_id(concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)  # 编码 concept，并保留 attention_mask
        token_count = max(int(concept_inputs.attention_mask.sum().item()) - 1, 0)  # 估计内容 token 数，减去最后的特殊 token
        token_indices[concept] = [0] if concept == "" and token_count == 0 else list(range(token_count))  # 空 anchor 特殊追踪第 0 位，否则追踪所有内容 token
    for concept in target_concepts:  # 再处理 target 文本
        concept_inputs = get_token_id(concept, pipeline.tokenizer_2, max_sequence_length, return_ids_only=False)  # 编码 target concept
        token_count = max(int(concept_inputs.attention_mask.sum().item()) - 1, 0)  # 计算 target 的内容 token 数
        token_indices[concept] = list(range(token_count))  # target 追踪所有内容 token 位置

    print("\nSelected text-side attention modules:")  # 打印本次选中的 text-side attention 模块标题
    for name in module_names:  # 遍历待编辑模块名
        print(f"  {name}")  # 打印当前模块名，方便检查编辑范围

    final_module_names = [final_modules[suffix] for suffix in selected_suffixes if suffix in final_modules]  # 按 Q/K/V 顺序取出各类最后模块名
    anchor_final_traces = _trace_many(pipeline, anchor_concepts, token_indices, final_module_names, args, device, max_sequence_length)  # 追踪 anchor 在最后模块上的输入输出
    anchor_final_means = {  # 保存每个最后模块对应的 anchor 平均输出
        module_name: _mean_outputs(anchor_final_traces, anchor_concepts, module_name)  # 对 anchor 输出沿样本维求平均
        for module_name in final_module_names  # 遍历每个最后模块
    }  # anchor 平均输出字典构造结束
    # endregion  # Target and Anchor 分段结束

    # region [Retain]  # 分段：追踪 retain 文本，用于估计保留概念的输入分布
    retain_traces = _trace_many(pipeline, retain_texts, token_indices, module_names, args, device, max_sequence_length)  # 对 retain 文本追踪所有待编辑模块
    retain_inputs_by_module = {}  # 保存每个模块对应的 retain 输入矩阵
    for module_name in module_names:  # 遍历每个待编辑模块
        retain_inputs = [  # 收集当前模块上所有 retain concept 的 inputs
            retain_traces[concept][module_name]["inputs"]  # 取出某个 retain concept 在当前模块的 inputs
            for concept in retain_texts  # 遍历所有 retain 文本
            if concept in retain_traces and module_name in retain_traces[concept]  # 只使用实际 trace 到该模块的 concept
        ]  # 当前模块 retain 输入列表结束
        if not retain_inputs:  # 如果当前模块没有任何 retain 输入
            raise RuntimeError(f"No retain trace for {module_name}")  # 直接报错，避免无法构造 retain 保护方向
        retain_inputs_by_module[module_name] = torch.cat(retain_inputs, dim=1)  # 将多个 retain 输入沿样本维拼成一个大矩阵
    # endregion  # Retain 分段结束

    edit_dict = {}  # 保存最终被编辑过的权重，用于写入 safetensors

    # region [Layer Update]  # 分段：逐个模块计算 delta 并直接更新权重
    for module_index, (module_name, module) in enumerate(edit_modules):  # 遍历每个待编辑模块及其序号
        suffix = module_suffixes[module_name]  # 获取当前模块的 Q/K/V 类型
        final_module_name = final_modules[suffix]  # 找到同类型的最后模块，用于衡量最终输出差异
        anchor_final_mean = anchor_final_means[final_module_name]  # 取出 anchor 在最后模块上的平均输出目标

        trace_module_names = list(dict.fromkeys([module_name, final_module_name]))  # 本轮只追踪当前模块和同类型最后模块，并去重
        edit_traces = _trace_many(pipeline, target_concepts, token_indices, trace_module_names, args, device, max_sequence_length)  # 追踪 target 在当前模型状态下的输入输出
        remaining_count = remaining_counts[module_index]  # 获取从当前模块开始还剩多少个同类模块，用于分摊 residual
        keys, residuals = [], []  # keys 保存当前模块输入，residuals 保存希望输出变化量
        for concept in target_concepts:  # 遍历每个 target 概念
            concept_trace = edit_traces[concept]  # 取出当前 target 的 trace 结果
            final_current = concept_trace[final_module_name]["outputs"]  # 取 target 当前在同类型最后模块上的输出
            target = anchor_final_mean.to(final_current.device, final_current.dtype).expand(-1, final_current.shape[1])  # 将 anchor 平均输出扩展到 target 输出样本数
            keys.append(concept_trace[module_name]["inputs"])  # 保存 target 在当前模块的输入，作为闭式解的 keys
            residuals.append((target - final_current) * (args.residual_scale / remaining_count))  # 计算当前模块应承担的目标输出差值

        keys = torch.cat(keys, dim=1).to(module.weight.device, torch.float32)  # 拼接所有 target keys，并转到当前权重设备的 float32
        residuals = torch.cat(residuals, dim=1).to(module.weight.device, torch.float32)  # 拼接所有 residuals，并转到当前权重设备的 float32
        retain_inputs = retain_inputs_by_module[module_name]  # 取当前模块对应的 retain 输入矩阵

        delta = _closed_form_update(keys,residuals,args.update_lambda,retain_inputs.to(module.weight.device, torch.float32),args.threshold,)  # 计算当前模块权重更新量 delta
        module.weight = torch.nn.Parameter(module.weight.float().add(delta).to(module.weight.dtype))  # 将 delta 加到当前模块权重上，并保持原 dtype
        edit_dict[module_name + ".weight"] = module.weight.detach().clone()  # 保存编辑后的权重副本到输出字典
        print(f"  Updated {module_name} | ||delta||={delta.norm().item():.4f}")  # 打印当前模块更新信息和 delta 范数
    # endregion  # Layer Update 分段结束

    if not edit_dict:  # 如果没有任何模块被编辑
        raise RuntimeError("No FLUX text-side attention weights were edited")  # 直接报错，说明编辑流程失败
    print(f"Current model status: Edited {target_concepts} into {anchor_concepts or ['null-anchor']}")  # 打印最终编辑状态
    return edit_dict  # 返回编辑后的权重字典


if __name__ == "__main__":  # 只有直接运行该脚本时才执行命令行入口
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # Base Config  # 分段：基础模型、保存路径和设备配置
    parser.add_argument("--sd_ckpt", help="base version for FLUX", type=str, default="black-forest-labs/FLUX.1-schnell")  # 指定 FLUX 基础模型路径或模型名
    parser.add_argument("--save_path", type=str, default=None)  # 指定编辑权重保存目录
    parser.add_argument("--file_name", type=str, default=None)  # 指定保存文件名，不含后缀
    parser.add_argument("--seed", type=int, default=0)  # 保留的随机种子参数，当前主流程未显式使用
    parser.add_argument("--device", type=str, default="cuda")  # 指定运行设备，如 cuda 或 cpu
    # Erase Config  # 分段：概念编辑相关配置
    parser.add_argument("--target_concepts", type=str, required=True)  # 指定要擦除/替换的目标概念，逗号分隔
    parser.add_argument("--anchor_concepts", type=str, required=True)  # 指定 anchor 概念，逗号分隔，空字符串表示 null-anchor
    parser.add_argument("--retain_path", type=str, default=None)  # 指定 retain 文本 CSV 路径
    parser.add_argument("--heads", type=str, default=None)  # 指定 CSV 中读取哪些列作为 retain_texts
    # Hyperparameters  # 分段：编辑超参数
    parser.add_argument("--params", type=str, default="KV", choices=["Q", "K", "V", "QK", "KV", "QKV"])  # 指定编辑 Q/K/V 中哪些 projection
    parser.add_argument("--threshold", type=float, default=1e-1)  # retain 零空间奇异值阈值
    # FLUX/MEMIT-specific controls  # 分段：FLUX trace 和闭式更新控制参数
    parser.add_argument("--trace_num_steps", type=int, default=4)  # trace 时运行的去噪步数
    parser.add_argument("--trace_seed", type=int, default=0)  # trace 时使用的随机种子
    parser.add_argument("--trace_resolution", type=int, default=512)  # trace 时使用的图像宽高
    parser.add_argument("--update_lambda", type=float, default=1e-4)  # 闭式解中的岭回归正则系数
    parser.add_argument("--residual_scale", type=float, default=1.0)  # residual 放大系数，控制编辑强度
    args = parser.parse_args()  # 解析命令行参数

    # 分段：解析 target 和 anchor 概念，并生成默认文件名后缀
    target_concepts = [con.strip() for con in args.target_concepts.split(",")]  # 将逗号分隔 target 字符串拆成列表并去掉首尾空格
    if not target_concepts or any(concept == "" for concept in target_concepts):  # 检查 target 列表是否为空或包含空概念
        raise ValueError("--target_concepts must not contain empty concepts")  # target 不能为空，因为空 target 没有明确编辑对象
    anchor_concepts = args.anchor_concepts  # 暂存原始 anchor 字符串
    retain_path = args.retain_path  # 暂存 retain CSV 路径

    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}"  # 用前 5 个 target 和 target 数量构造默认文件名后缀
    anchor_concepts = [x.strip() for x in anchor_concepts.split(",")]  # 将逗号分隔 anchor 字符串拆成列表并去掉首尾空格
    if len(anchor_concepts) == 1:  # 如果只给了一个 anchor
        anchor_concepts = anchor_concepts * len(target_concepts)  # 将同一个 anchor 复制给所有 target
        if anchor_concepts[0] == "":  # 如果 anchor 是空字符串
            file_suffix += "-to_null"  # 文件名标记为编辑到 null-anchor
        else:  # 如果 anchor 不是空字符串
            file_suffix += f"-to_{anchor_concepts[0]}"  # 文件名标记为编辑到该 anchor
    else:  # 如果给了多个 anchor
        if len(target_concepts) != len(anchor_concepts):  # 检查 target 和 anchor 是否一一对应
            raise ValueError("target_concepts and anchor_concepts must have the same length")  # 数量不一致则报错
        file_suffix += f"-to_{anchor_concepts[0]}_etc"  # 多 anchor 文件名只显示第一个并加 etc

    # 分段：读取 retain 文本
    retain_texts = []  # 初始化 retain 文本列表
    if retain_path is not None:  # 如果用户提供了 retain CSV 文件
        if not retain_path.endswith(".csv"):  # 检查 retain 文件后缀
            raise ValueError("--retain_path must be a .csv file")  # 当前只支持 CSV 文件
        if args.heads is None:  # 如果使用 retain CSV 但没有指定列名
            raise ValueError("--heads is required when --retain_path is used")  # 要求显式指定读取哪些列
        df = pd.read_csv(retain_path)  # 读取 retain CSV 为 DataFrame
        for head in args.heads.split(","):  # 遍历逗号分隔的列名
            retain_texts += df[head.strip()].unique().tolist()  # 取该列唯一值并加入 retain_texts
    else:  # 如果没有提供 retain CSV
        retain_texts.append("")  # 使用空字符串作为默认 retain 文本

    # 分段：加载模型、执行编辑并保存结果
    save_path = args.save_path or "logs/checkpoints"  # 使用用户保存目录，或默认 logs/checkpoints
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"  # 使用用户文件名，或用时间戳和后缀生成文件名
    max_sequence_length = 256 if "schnell" in args.sd_ckpt else 512  # schnell 模型使用 256，其它模型使用 512

    pipeline = DiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=torch.bfloat16).to(args.device)  # 加载 FLUX pipeline，并转到指定设备
    edit_dict = edit_model(  # 调用主编辑函数，返回被修改的权重字典
        args=args,  # 传入命令行参数对象
        pipeline=pipeline,  # 传入已加载的 FLUX pipeline
        target_concepts=target_concepts,  # 传入目标概念列表
        anchor_concepts=anchor_concepts,  # 传入 anchor 概念列表
        retain_texts=retain_texts,  # 传入 retain 文本列表
        device=args.device,  # 传入运行设备
        max_sequence_length=max_sequence_length,  # 传入最大文本序列长度
    )  # edit_model 调用结束
    os.makedirs(save_path, exist_ok=True)  # 创建保存目录，如果已存在则不报错
    save_file(edit_dict, os.path.join(save_path, f"{file_name}.safetensors"))  # 将编辑后的权重保存为 safetensors 文件
