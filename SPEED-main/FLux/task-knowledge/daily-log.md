# Daily Log

## 2026-06-29

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

## 2026-06-30

日期：
2026-06-29

今日目标：
将 FLUX 概念擦除代码整理为接近 `train_erase_null.py` 的单文件训练/编辑脚本结构，同时保留 MEMIT-style 机制。当前目标不是追求最优擦除效果，而是把 MEMIT 的多层真实激活写入机制完整迁移到 FLUX dual block 文本侧 q/k 权重编辑中。

今日完成：
- 明确权重编辑位置限定为 FLUX dual block 的文本侧 q/k：`transformer_blocks.{i}.attn.add_q_proj.weight` 和 `transformer_blocks.{i}.attn.add_k_proj.weight`。
- 排除图像侧 q/k/v、文本侧 v、single transformer blocks、context embedder 和其他非目标模块，避免污染变量和编辑范围。
- 将代码结构整理，包括 token 定位、目标模块筛选、forward trace、闭式更新、`edit_model(...)` 主入口、CLI 参数解析和 safetensors 保存。
- 保留 MEMIT-style 的核心机制：通过真实 FLUX forward hook 获取每个目标层的实际输入/输出状态，而不是用静态文本 embedding 代替中间状态。
- 将概念参数整理为 `target_concepts`、`anchor_concepts`、`retain_concepts`，并保留旧参数别名用于兼容。
- 在更新公式中保留原权重与闭式求得的增量结合，即 `W_new = W_old + delta`。

修改文件：
- `SPEED-main/FLux/CE_Flux.py`
- `SPEED-main/FLux/sample.py`


验证结果：
- 目前尚未完成真实 FLUX pipeline/GPU 运行验证；现阶段验证只覆盖文件结构、静态代码逻辑和日志写入。

当前问题：
- 默认编辑层范围仍未经过实验验证，不能假设中层间隔选择就是最优。
- q/k 编辑是否足以稳定完成 concept eraser 任务，需要通过生成结果和保留概念评估确认。
- `retain_threshold`、`update_lambda`、`residual_scale` 等超参数仍需要实验调参。
- anchor 为空时使用 null/retain 的策略需要进一步统一，否则不同实验之间不可比。

下一步计划：
- 先运行最小 smoke test：单个 target、单个 anchor、单层 q/k、低分辨率 trace。
- 检查生成的 `.safetensors` 是否只包含预期的文本侧 q/k 权重 key。
- 用 `sample.py` 加载编辑权重，比较 original 和 erased 输出。
- 在最小流程确认可运行后，再扩展到多层、多概念和 retain 消融实验。

需要人工确认的地方：
- anchor 为空时，是使用 null-anchor、retain 均值，还是必须显式指定 anchor concept。
- 后续效果验证采用人工看图、CLIP 相似度。

！实验方法问题：FLUX 概念编辑中，token pooling 方式可能影响效果：mean pooling 更稳健，last-token pooling 更接近 SPEED，后续需对名人/风格删除做消融实验。

  CUDA_VISIBLE_DEVICES=0 python FLux/CE_Flux.py \
  --target_concepts "Van Gogh" \
  --anchor_concepts "painting" \
  --retain_path "FLux/data/style_100.csv" \
  --heads "concept" \
  --save_path "FLux/models" \
  --file_name "erase_vangogh_to_painting_QK_r10" \
  --params QK \
  --residual_scale 10.0 \
  --update_lambda 1e-3 \
  --threshold 1e-1

  CUDA_VISIBLE_DEVICES=0 python FLux/sample.py \
  --mode original,edit \
  --erase_type style \
  --target_concept "Van Gogh" \
  --contents "Van Gogh" \
  --edit_ckpt "FLux/models/erase_vangogh_to_painting_QK_r10.safetensors" \
  --save_root "FLux/results_vangogh_to_painting_QK_r10" \
  --prompts "a painting by {}; a landscape by {};  a village scene by {}; a flower vase by {}" \
  --num_samples 20 \
  --batch_size 5 \
  --strict_edit_load




  CUDA_VISIBLE_DEVICES=0 python FLux/CE_Flux.py \
  --target_concepts "Snoopy" \
  --anchor_concepts "" \
  --retain_path "FLux/data/instance_small.csv" \
  --heads "concept" \
  --save_path "FLux/models" \
  --file_name "erase_snoopy_to_dog_QKV_r8" \
  --params QKV \
  --residual_scale 8.0 \
  --update_lambda 1e-3 \
  --threshold 1e-1

  CUDA_VISIBLE_DEVICES=0 python FLux/sample.py \
  --mode original,edit \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Snoopy" \
  --edit_ckpt "FLux/models/erase_snoopy_to_dog_QKV_r8.safetensors" \
  --save_root "FLux/results_snoopy_to_null_QKV_r8" \
  --prompts "a photo of {}; {} in a park; {} character" \
  --num_samples 20 \
  --batch_size 5 \
  --strict_edit_load

  CUDA_VISIBLE_DEVICES=0 python FLux/sample.py \
  --mode original,edit \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Hello Kitty" \
  --edit_ckpt "FLux/models/erase_snoopy_to_dog_s5.safetensors" \
  --save_root "FLux/results_retain_hellokitty_after_snoopy_s5_fresh" \
  --prompts "a photo of {}; a cartoon image of {}; {} character; {} in a park; {} with a colorful background" \
  --num_samples 20 \
  --batch_size 5 \
  --strict_edit_load