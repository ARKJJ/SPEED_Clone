# Current Task

## 任务日期

2026-06-28

## 昨日任务目标

昨日任务是将 MEMIT 的层差偏移机制结合到 FLUX 中，形成一个初步的 FLUX 概念擦除编辑脚本。具体来说，不是继续沿用 Stable Diffusion 中编辑 UNet cross-attention 的路径，而是面向 FLUX transformer 结构，尝试在文本条件相关的 q/k 投影层中写入概念偏移。

这个目标重要，是因为 FLUX 的结构和 Stable Diffusion v1.4 不同。若直接照搬 SPEED 原始实现中对 UNet `to_k/to_v` 的编辑位置，理论上并不能保证作用在 FLUX 的关键语义通道上。MEMIT 的优势在于它不只依赖静态权重名，而是先追踪中间层输入/输出，再用“当前输出到目标输出的差值”构造可写入权重的偏移。

## 已完成内容

昨日已经在 `CE_Flux.py` 中搭建了 MEMIT-style FLUX 编辑流程：

- 定义 `MemitFluxConfig`，集中管理层范围、trace 步数、分辨率、正则强度和残差缩放。
- 使用 `DiffusionPipeline.from_pretrained` 加载 FLUX 模型，默认模型为 `black-forest-labs/FLUX.1-schnell`。
- 通过 `_select_text_qk_modules` 选择 `transformer_blocks.*.attn.add_q_proj` 和 `add_k_proj`。
- 通过 `_trace_prompt` 注册 forward pre-hook 和 forward hook，记录选定模块的输入和输出。
- 对 `anchor_concepts` 计算目标输出均值。
- 暂不加入 retain Gram 正则，避免把额外保留约束和 MEMIT 层差偏移主机制混在一起。
- 对 `target_concepts` 计算当前输出与目标输出之间的 residual。
- 通过 `_closed_form_update` 求解每个模块的权重偏移 `delta`。
- 将 `delta` 加回模块权重，并把编辑后的局部权重保存为 `.safetensors`。

## 当前实现的核心判断

当前实现选择编辑文本侧 q/k 投影层。这一选择的理由是：概念擦除本质上是改变文本条件如何影响生成过程，而 FLUX transformer 中的 attention q/k 投影直接参与 token 与隐空间表示的匹配关系。通过修改这些层，有可能改变模型对目标概念 token 的响应方式。

但这个判断还只是工作假设，不能当作结论。是否只编辑 q/k 就足够，是否需要加入 v/o 或其他模块，必须通过采样实验和消融验证。

## 需要继续修改的模块

优先级最高的是 FLUX 版采样验证模块。当前 `CE_Flux.py` 已经能保存 `.safetensors` 局部权重，但还缺少一个可靠脚本将这些权重加载回 FLUX pipeline，并在同 prompt、同 seed 下比较 original 和 edited 输出。

其次需要整理 `scripts/`。当前脚本仍引用 `train_erase_null.py`、`sample2.py` 和 `src/...`，与 FLUX 当前实现不一致。它们不能直接作为 FLUX 评估脚本使用。

还需要补一个 tokenizer 检查工具。`--replace_indices` 对 MEMIT-style 编辑很关键，如果不知道 FLUX tokenizer 如何切分目标概念，就无法判断 residual 是否对准了正确 token。

## 验证方式



最小运行验证：

```bash
cd /Users/ark/SPEED_Clone/SPEED-main/FLux
python3 CE_Flux.py \
  --concept_type object \
  --target_concepts "Snoopy" \
  --anchor_concepts "" \
  --retain_concepts "Mickey;Hello Kitty" \
  --trace_num_steps 1 \
  --layer_start 6 \
  --layer_end 8 \
  --save_dir "./models" \
  --exp_name "smoke_snoopy"
```

## 待确认

- 实现是否已经在 GPU 上完整跑通过。
- `models/*.safetensors` 是否已有实际输出。
- q/k 编辑是否足够，还是需要扩展到更多 FLUX attention 模块。
- `anchor_concepts` 为空时使用 `retain_concepts` 作为 anchor 来源是否合理。
- 现有 `sample.py` 和 `scripts/` 是否计划迁移到 FLUX，还是应另写新脚本。
