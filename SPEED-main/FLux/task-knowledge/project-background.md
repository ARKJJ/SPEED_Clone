# Project Background

## 项目定位

本项目当前关注的是 FLUX 概念擦除实验。它的目标不是复述原始 SPEED 仓库的 Stable Diffusion v1.4 流程，而是把 SPEED 中“快速、精确地编辑扩散模型概念”的研究动机，与 MEMIT 的层内闭式更新思想结合起来，尝试迁移到 FLUX 模型结构中。

这一区分很重要。SPEED 原论文主要围绕 Stable Diffusion 的 UNet cross-attention 权重做概念擦除；而当前 FLUX 代码编辑的是 FLUX transformer block 中和文本条件相关的 q/k 投影层。如果把这两者混为一谈，就会误以为 FLUX 版本已经继承了原始 SPEED 的所有理论和实验结论。这个判断是不严谨的：当前 FLUX 版本还需要单独验证。

## 研究目标

当前 FLUX 任务的核心目标是：在不完整微调 FLUX 的前提下，通过对若干 transformer 层进行局部权重编辑，使模型对指定目标概念的响应发生偏移，同时尽量保持保留概念的生成能力。

可以把目标拆成三层：

1. 概念擦除：让 `target_concepts` 对应的目标概念弱化或被引导到更泛化的语义。
2. 语义引导：通过 `anchor_concepts` 指定目标概念应偏移到的方向，例如将具体艺术家风格引导到一般 `art`。
3. 保留约束：通过 `retain_concepts` 构建保留语义，避免更新破坏非目标概念。

这些信息重要，是因为概念擦除不是简单地“让模型生成失败”。真正有研究价值的擦除应当是定向的：目标概念变弱，非目标概念尽量稳定，整体图像质量不能明显崩坏。

## 与 SPEED 的关系

SPEED 提供了项目的研究背景：概念擦除应当同时追求 scalable、precise、efficient。它说明了为什么不能只依赖慢速 fine-tuning，也不能接受粗暴删除导致的大范围副作用。

当前 FLUX 项目借用了这个问题意识，但实现路径不同：

- SPEED 的 Stable Diffusion 版本主要编辑 UNet cross-attention 的 `to_k/to_v` 权重。
- 当前 FLUX 版本在 `CE_Flux.py` 中选择 `transformer_blocks.*.attn.add_q_proj` 和 `add_k_proj`。
- SPEED 使用 target、anchor、retain 等语义约束构造闭式更新。
- 当前 FLUX 版本更接近 MEMIT-style：先 trace 某些层的输入/输出，再根据目标输出差异求解权重偏移。

因此，准确说法应是：当前项目是一个 SPEED-inspired 的 FLUX 概念擦除实现，同时引入 MEMIT 的层差偏移机制。

## 与 MEMIT 的关系

MEMIT 的核心思想可以概括为：通过追踪模型中间层的激活，计算当前事实或概念表征与目标表征之间的残差，再将这个残差以闭式解的形式写入若干层权重。当前 `CE_Flux.py` 中的 `_trace_prompt`、`_closed_form_update` 和逐层 `delta` 更新就是这种思路在 FLUX 上的工程化尝试。

这对 FLUX 很关键，因为 FLUX 不是 Stable Diffusion v1.4 的 UNet 架构。直接照搬 SPEED 对 UNet cross-attention 的编辑位置并不成立。通过 MEMIT 式 trace，可以先观察 FLUX transformer 中某些层对文本概念的实际输入/输出，再把目标偏移写入对应权重。

## 当前整体方法

当前 FLUX 方法在代码层面分为以下步骤：

1. 加载 FLUX pipeline，默认模型为 `black-forest-labs/FLUX.1-schnell`。
2. 根据 `layer_start`、`layer_end`、`layer_stride` 选择 transformer block。
3. 从这些 block 中选择文本侧 q/k 投影模块：`.attn.add_q_proj` 和 `.attn.add_k_proj`。
4. 对 anchor concepts 运行短步数生成，记录目标模块输出，形成目标输出均值。
5. 对 retain concepts 运行 trace，使用其模块输入构建 preserve Gram 约束。
6. 对 target concepts 运行 trace，计算当前输出到 guide 输出之间的 residual。
7. 使用闭式线性求解得到每个模块的 `delta`。
8. 将 `delta` 加到原模块权重，并把编辑后的权重保存为 `.safetensors`。

## 当前限制

- 当前方法是否真的能稳定擦除 FLUX 中的目标概念，仍需采样验证。

- `scripts/` 中脚本仍引用 `train_erase_null.py`、`sample2.py`、`src/...` 等当前 FLux 目录未包含的文件，是否已经完成迁移待确认。
- `CE_Flux.py` 保存的是部分 transformer 权重，不是完整 FLUX pipeline。
- `preserve_concepts` 的选择会显著影响编辑副作用，目前缺少固定协议。


