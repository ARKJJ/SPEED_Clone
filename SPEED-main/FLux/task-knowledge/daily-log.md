# Daily Log

## 2026-06-28

### 完成

- 将 MEMIT 的层差偏移机制放入 FLUX 编辑流程，形成 `CE_Flux.py` 的初步实现。
- 增加 `MemitFluxConfig`，集中配置编辑层范围、trace 步数、trace seed、trace 分辨率、preserve 正则和 residual 缩放。
- 在 FLUX transformer 中选择文本侧 q/k 投影层，即 `.attn.add_q_proj` 和 `.attn.add_k_proj`。
- 使用 forward hook 追踪 edit、guide、preserve concepts 在选定层的输入输出。
- 使用 guide 输出均值作为目标方向，使用 edit 当前输出与目标输出之间的差值作为 residual。
- 通过闭式线性求解得到每层 `delta`，并将编辑后的局部权重保存为 `.safetensors`。

### 为什么重要

工作的关键意义在于：它把 FLUX 概念擦除从“照搬 Stable Diffusion 的 UNet 编辑位置”推进到“根据 FLUX 自身 transformer 层激活来写入权重偏移”。这更符合 FLUX 架构，也为后续做层选择、模块选择和 preserve 约束消融提供了基础。

### 观察

- 当前实现更接近 MEMIT-style 层编辑，而不是 SPEED 原始 Stable Diffusion 公式的逐项复现。
- 默认编辑层为 `layer_start=6` 到 `layer_end=15`，步长为 `2`，该范围尚未经过实验确认。

### 待确认

- 是否已经成功生成 `models/*.safetensors`。
- q/k 层差偏移是否能稳定削弱目标概念。

