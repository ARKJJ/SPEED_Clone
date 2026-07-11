# Project Background

## 项目定位

当前 `FLux/` 子项目关注的是 FLUX 概念擦除和概念迁移实验。它不是原始 SPEED Stable Diffusion v1.4 流程的直接复刻，而是一个 SPEED-inspired、MEMIT-style 的 FLUX 局部权重编辑实现。

核心问题是：在不完整微调整个 FLUX 模型的前提下，是否可以只改一部分 transformer attention 投影权重，让模型对指定目标概念的响应发生定向偏移，同时尽量保留其他概念的生成能力。

## 与 SPEED 的关系

SPEED 提供研究动机：概念擦除应尽量 scalable、precise、efficient。原始 SPEED 主要围绕 Stable Diffusion 的 UNet cross-attention 权重展开。

当前 FLUX 实现不同：

- 模型结构是 FLUX transformer，不是 SD v1.4 UNet。
- 编辑位置是 `transformer_blocks.*.attn.add_q_proj`、`add_k_proj`、`add_v_proj` 中由 `--params` 指定的文本侧 attention 投影层。
- 输出是局部 transformer 权重 `.safetensors`，不是完整 pipeline。
- 当前方法需要单独通过 FLUX 采样验证，不能直接继承 SPEED 论文中的实验结论。

## 与 MEMIT 的关系

当前 `CE_Flux.py` 更接近 MEMIT 的工程思路：

1. 运行真实 FLUX forward。
2. 用 hook 收集目标模块的输入和输出。
3. 计算 target 当前输出到 anchor 输出均值之间的 residual。
4. 使用 retain 输入构造保护子空间。
5. 用闭式线性求解得到权重增量 `delta`。
6. 把 `delta` 写回对应 attention 投影层权重。

这个流程的价值在于它依赖真实中间激活，而不是只根据静态权重名做编辑。

## 当前整体方法

当前 `CE_Flux.py` 的主流程：

1. 设置 Hugging Face 镜像端点。
2. 加载 `DiffusionPipeline.from_pretrained(sd_ckpt, torch_dtype=torch.bfloat16)`。
3. 按 `--params` 在 `transformer_blocks.*` 中选择 Q/K/V 文本侧 attention 投影层。
4. 使用 `tokenizer_2` 为 target、anchor、retain 文本定位有效 token。
5. 对 anchor 概念在每类 suffix 的最后模块处收集输出均值。
6. 对 retain 文本在所有编辑模块处收集输入，构造 retain 输入矩阵。
7. 对 target 概念在当前模块和最终模块处收集 trace。
8. 将 target 当前输出拉向对应 anchor 输出均值，并按剩余同类模块数分摊 residual。
9. 用 retain 零空间投影和 ridge 正则求解 `delta`。
10. 保存编辑后的局部权重。

## 当前配置原则

必须显式记录每次实验的以下配置：

- `target_concepts`
- `anchor_concepts`
- `retain_path`
- `heads`
- `params`
- `trace_num_steps`
- `trace_seed`
- `trace_resolution`
- `update_lambda`
- `threshold`
- `residual_scale`
- `sd_ckpt`

其中 `params` 尤其重要，因为 `KV`、`QK`、`QKV` 对编辑强度和副作用可能完全不同。

## 当前限制

- `CE_Flux.py` 已通过语法检查，但尚未在知识库中记录完整 GPU 运行结果。
- 当前模块选择覆盖所有匹配的 `transformer_blocks`，还没有 layer 范围消融。
- 生成效果、retain 副作用、最佳 `params` 组合都仍需实验验证。

## 推荐实验顺序

1. 先用 `instance_small.csv` 或 `style_100.csv` 做单 target、单 anchor、小规模 retain 的 smoke test。
2. 检查输出 `.safetensors` 是否只含预期模块 key。
3. 修复并验证 `sample.py`，确保同 seed original/edit 可对照。
4. 比较 `KV`、`QK`、`QKV`。
5. 再考虑增加 layer 范围参数，做层选择消融。
