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
- 后续效果验证采用人工看图、CLIP 相似度。

```bash
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
  --file_name "erase_snoopy_to_null_KV_r8" \
  --params KV \
  --residual_scale 8.0 \
  --update_lambda 1e-3 \
  --threshold 1e-1

CUDA_VISIBLE_DEVICES=0 python FLux/sample.py \
  --mode original,edit \
  --erase_type instance \
  --target_concept "Snoopy" \
  --contents "Snoopy" \
  --edit_ckpt "FLux/models/erase_snoopy_to_null_KV_r8.safetensors" \
  --save_root "FLux/results_snoopy_to_null_KV_r8" \
  --num_samples 2 \
  --batch_size 5 
  

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
```

## 2026-07-04

### 阶段目标

这一阶段的重点是重构闭式解公式，让 FLUX 版编辑不只是简单地把 target 输出拉向 anchor，而是把 retain 约束更明确地写进权重更新空间。也就是说，核心工作从“能算出 delta”推进到“delta 应该落在哪个子空间、如何减少对 retain 概念的副作用”。

### 进展

- 重构 `_closed_form_update` 的计算方式，引入 retain 输入协方差矩阵。
- 使用 retain 输入的 SVD 结果构造近似零空间，将更新限制到更不影响 retain 概念的方向。
- 将 `threshold` 明确为选择 retain 零空间方向的阈值，而不是普通的经验开关。
- 保留 ridge 项 `update_lambda`，用于稳定闭式线性系统求解。
- 将 residual 按剩余同类模块数分摊，避免每一层都一次性承担完整输出偏移。
- 将 retain 文本来源固定为 `--retain_path` 加 `--heads`，便于后续重跑实验时保持相同保护集合。

### 观察

- 闭式解的效果对 `threshold` 很敏感：阈值越大，可更新子空间越小，retain 保护可能更强，但目标擦除也可能变弱。
- `update_lambda` 过小时容易放大不稳定方向，过大时会压低编辑强度。
- `residual_scale` 实际上承担了“擦除强度旋钮”的作用，需要和 `params`、`threshold` 联合调。

### 待继续

- 用相同 target/anchor/retrain 配置重跑实验，比较重构闭式解前后的生成效果。
- 对 `QK`、`KV`、`QKV` 分别调参，观察擦除强度和 retain 副作用。
- 优先记录 Van Gogh/style 和 Snoopy/instance 两类实验，因为它们分别代表风格擦除和对象擦除。

## 2026-07-07

### 阶段目标

这一阶段的重点是围绕重构后的闭式解重新跑实验，并根据生成结果调整参数。目标不是一次性确定最优配置，而是建立一组可比较的实验：同一 target/anchor/retain 下，只改变 `params`、`residual_scale`、`update_lambda` 或 `threshold`。

### 进展

- 重新跑了风格类实验，例如将 `Van Gogh` 引导到 `painting`，并使用 `style_100.csv` 作为 retain 集合。
- 重新跑了对象类实验，例如 `Snoopy` 的 null-anchor/object 擦除，并使用 `instance_small.csv` 作为 retain 集合。
- 尝试提高 `residual_scale`，例如 `r9`、`r10` 这类配置，用于增强目标概念削弱效果。
- 尝试把 `update_lambda` 调到 `1e-3`，增强线性系统的数值稳定性。
- 对 `QK` 和 `QKV` 等模块组合进行比较，观察更强编辑范围是否带来更明显擦除和更多副作用。
- 对 retain 概念单独采样，例如在 Snoopy 编辑后检查 Hello Kitty，观察非目标概念是否被破坏。

### 当前问题

- 参数之间存在耦合：`QKV + 高 residual_scale` 可能更强，但 retain 副作用也更需要关注。
- 风格擦除和对象擦除对参数的敏感性可能不同，不能直接共用同一组超参数结论。

### 下一步

- 固定一组最小 benchmark：Van Gogh/style、Snoopy/instance、一个 retain concept 对照。
- 把每次实验的 `params`、`residual_scale`、`update_lambda`、`threshold` 写入文件名或日志。
- 根据重跑结果决定默认配置是否从 `QK` 转向 `KV` 或 `QKV`。
- 保留 `sample.py` 加载链路检查

## 2026-07-11

### 阶段目标

在当前 FLUX 版 `CE_Flux.py` 中补入 SPEED 的 IPF 思想：先精炼 retain set，再用精炼后的 retain inputs 构造零空间 projector。实现要求是尽量贴近 SPEED 原版做法，同时避免把逻辑拆成过多新函数。

### 完成内容

- 在 `CE_Flux.py` 的 `_closed_form_update(...)` 内部加入 IPF retain refinement。
- 默认启用 IPF；新增 `--disable_filter`，用于关闭 IPF 并回退到原来的全 retain set 零空间构造路径。
- IPF 打分方式与 SPEED 原版保持同构：
  - SPEED SD 版使用 `ret_embs @ erase_weight.T` 的范数筛 retain。
  - FLUX 版没有同一个静态 `erase_weight`，因此先用当前 `keys/residuals` 估计未投影的初始编辑方向 `delta_init`。
  - 再用 `(delta_init @ retain_inputs).norm(dim=0)` 作为 retain token 影响分数。
  - 保留 `score > score.mean()` 的 retain inputs，用于后续 covariance/SVD/null-space projector。
- 为防止极端情况下所有 score 相同导致筛空，加入轻量保护：如果 `keep_mask` 为空，则保留原 retain set。
- 将 IPF 初始方向的求解从 hidden 维度大矩阵 solve 改为等价 dual form，只解 token/sample 维度小矩阵，减少 FLUX 多层编辑时的额外开销。

### 修改文件

- `SPEED-main/FLux/CE_Flux.py`

### 验证

`_closed_form_update` 的两个行为：
  - 默认路径会使用 IPF 精炼后的 retain set。
  - `disable_filter=True` 时保持原来的 unfiltered retain 行为。


### 尚未验证

- 尚未运行真实 FLUX pipeline/GPU 编辑与采样验证。
- 后续仍需要用固定 target/anchor/retain 配置比较启用 IPF 与 `--disable_filter` 的生成效果、retain 副作用和运行耗时。

## 2026-07-14

### 阶段目标

梳理 FLUX 版 `CE_Flux.py` 中 T5 文本 token 的选择策略，先做一个更接近 SPEED 思路的消融：普通概念不再默认使用全体有效 token，而是用单个主体 token 表示概念，从而减少 trace token 数、闭式解样本列数和 retain covariance 的计算量。

### 完成内容

- 讨论了 CLIP 和 T5 encoder 的差异：CLIP text transformer 的 causal attention 让后方 token 更容易聚合前文信息，而 T5 encoder 是双向 attention，每个 token 都能看全句，因此最后 token 没有天然汇聚优势。
- 将 `target_concepts` 和 `anchor_concepts` 放在同一个 paired loop 中选择 token。
- 普通 `anchor_concepts` 的 token 选择改为首个有效 token：
  - 使用 `int(concept_inputs.attention_mask[0].argmax().item())` 从 `attention_mask` 中定位第一个有效 token。
- 将非空 `retain_texts` 的 token 选择改为首个有效 token。
- 保留空 retain 的原始行为：
  - `concept == ""` 时使用 `list(range(0, concept_inputs.input_ids.shape[1]))`，避免 T5 无 BOS 时跳过 index 0 的 null token。
  - 这样 null/empty preserve 路径暂时不引入新的实验变量。
- 将普通 `target_concepts` 的 token 选择改为首个有效 token。
- 保留 `target_concepts == ["nudity"]` 的 SPEED-style 全 token span 行为：
  - `nudity` 时 target 和 anchor 都使用 `list(range(0, concept_inputs.input_ids.shape[1]))`，避免 T5 无 BOS 时跳过 index 0 的主体 token。
  - 这样 target/anchor 的 trace 列数保持一致，后续可以逐 token/逐 step 计算 residual，避免 target 多 token 而 anchor 单 token 的 shape mismatch。


### 修改文件

- `SPEED-main/FLux/CE_Flux.py`

### 当前 token 选择规则

- anchor：普通概念取首个有效 token；`nudity` 分支下与 target 一起取全 token span。
- retain 非空文本：首个有效 token。
- retain 空文本：沿用原来的多 token/null retain 逻辑。
- target 普通概念：首个有效 token。
- target 为 `nudity`：target 和 anchor 都沿用全 token span 逻辑。

### 验证

- 已运行 `python -m py_compile FLux/CE_Flux.py`，语法检查通过。

### 待验证

- 仍需运行真实 FLUX pipeline/GPU 编辑，比较首个主体 token 与原先最后主体 token/多 token 策略下的擦除强度、retain 副作用和运行耗时。
CUDA_VISIBLE_DEVICES=0 python CE_Flux.py \
  --sd_ckpt "black-forest-labs/FLUX.1-dev" \
  --device "cuda:0" \
  --target_concepts "Snoopy" \
  --anchor_concepts "" \
  --retain_path "data/instance_small.csv" \
  --heads "concept" \
  --save_path "logs/checkpoints" \
  --file_name "erase_snoopy_to_null_V_r1" \
  --params V \
  --trace_num_steps 20 \
  --residual_scale 1

CUDA_VISIBLE_DEVICES=1 python FLux/sample.py \
  --sd_ckpt "black-forest-labs/FLUX.1-dev" \
  --mode "original,edit" \
  --edit_ckpt "FLux/logs/checkpoints/erase_snoopy_to_null_V_r4_t10.safetensors" \
  --save_root "FLux/logs/FLUX/instance_KV4" \
  --erase_type "instance" \
  --target_concept "Snoopy" \
  --contents "Snoopy" \
  --num_samples 2 \
  --batch_size 4

CUDA_VISIBLE_DEVICES=0 python FLux/sample.py \
  --sd_ckpt "black-forest-labs/FLUX.1-dev" \
  --mode "original,edit" \
  --edit_ckpt "FLux/logs/checkpoints/erase_snoopy_to_null_V_r5_t10.safetensors" \
  --save_root "FLux/logs/FLUX/instanceV5" \
  --erase_type "instance" \
  --target_concept "Snoopy" \
  --contents "Mickey, Spongebob, Pikachu, Hello Kitty" \
  --num_samples 4 \
  --batch_size 4

CUDA_VISIBLE_DEVICES=1 python FLux/CE_Flux.py \
  --sd_ckpt "black-forest-labs/FLUX.1-dev" \
  --device "cuda:0" \
  --target_concepts "Van Gogh" \
  --anchor_concepts "painting" \
  --retain_path "FLux/data/style_100.csv" \
  --heads "concept" \
  --save_path "FLux/logs/checkpoints" \
  --file_name "erase_vangogh_to_painting_V_r8_t10" \
  --params V \
  --trace_num_steps 10 \
  --residual_scale 8

  CUDA_VISIBLE_DEVICES=1 python FLux/sample.py \
  --sd_ckpt "black-forest-labs/FLUX.1-dev" \
  --mode "original,edit" \
  --edit_ckpt "FLux/logs/checkpoints/erase_vangogh_to_painting_V_r8_t10.safetensors" \
  --save_root "FLux/logs/FLUX/styleV8" \
  --erase_type "style" \
  --target_concept "Van Gogh" \
  --contents "Van Gogh" \
  --num_samples 2 \
  --batch_size 4

  ### 目前已测试参数
  10个timestep联合约束
  instanace erasure
  QK r=10 snoopy擦除效果弱
  KV r=10 snoopy擦除效果一般
  QKV r=10 snoopy在部分prompt下表现差，擦除效果不明显
  V r=10 snoopy擦除效果明显

  style erasure
  KV r=5 擦除效果可以   r=10 画面退化严重
  

## 2026-07-15

### 阶段目标

围绕 FLUX concept erasure 建立一组可复现的参数扫描脚本。当天重点不是确定最终最优超参数，而是先把“编辑权重、采样图片、计算指标”的流程固定下来，方便后续比较 V、KV、QKV 以及不同 `residual_scale` 的影响。

### 完成内容

- 创建 V-only 实例擦除脚本：
  - 脚本：`FLux/scripts/run_concept_erasure_eval_V.sh`
  - GPU：`CUDA_VISIBLE_DEVICES=0`
  - 参数组：`--params V`
  - `residual_scale` 扫描范围：`r=7..15`
  - checkpoint 命名：`erase_snoopy_to_null_V_r${r}_t10.safetensors`
  - 图片输出目录：`FLux/logs/FLUX/instanceV${r}`
- 创建 KV 实例擦除脚本：
  - 脚本：`FLux/scripts/run_concept_erasure_eval_KV.sh`
  - GPU：`CUDA_VISIBLE_DEVICES=1`
  - 参数组：`--params KV`
  - `residual_scale` 扫描范围：`r=7..15`
  - 图片输出目录：`FLux/logs/FLUX/instanceKV${r}`
- 创建 QKV 实例擦除脚本：
  - 脚本：`FLux/scripts/run_concept_erasure_eval_QKV.sh`
  - 参数组：`--params QKV`
  - `residual_scale` 扫描范围：`r=7..10`
  - 图片输出目录：`FLux/logs/FLUX/instanceQKV${r}`
- 统一采样目录命名规则：
  - 将 `instanceV_r10_t10` 这类目录改为更紧凑的 `instanceV10`、`instanceKV10`、`instanceQKV10`。
  - checkpoint 文件名仍保留 `V/KV/QKV`、`r` 和 `t10` 信息，便于追踪实验配置。
- 在 `FLux/score_cal.py` 中加入 baseline CS：
  - `Edit CS` 表示 edit 图片与 prompt 的 CLIP Score。
  - `Baseline CS` 表示 original 图片与 prompt 的 CLIP Score。
  - `FID` 仍然表示 edit 图片与对应 original 图片之间的 FID。
- 明确当前 FID 口径：
  - 当前 FLUX 实例脚本计算的是同一次实验中生成的 `edit` 图片和 `original` 图片之间的 FID。
  - 这不是和真实图片数据集比较的 FID，也不是严格复现论文中的 FID protocol。
  - 默认 `num_samples=2` 时，每个 instance concept 大约有 `80 * 2 = 160` 张图，FID 可以用来看粗略趋势，但绝对数值方差较大。
- 创建 SPEED 轻量实例擦除 FID 脚本：
  - 脚本：`SPEED-main/scripts/run_instance_fid_light_SPEED.sh`
  - 目标：快速观察原版 SPEED 实例擦除时 edit/original FID 的量级。
  - 默认目标：`Snoopy`
  - 默认采样：`num_samples=1`，只作为快速 sanity check。

### 参数记录与观察

- V-only 在早期单概念 Snoopy 擦除检查中效果比较明显，因此优先做 `V, r=7..15` 的较宽扫描。
- KV 被放到 GPU1 上作为并行对照组，用于观察 K/V 联合编辑是否比 V-only 更稳定。
- QKV 只先跑 `r=7..10`，因为它编辑的参数范围更大，潜在副作用也更强。
- 当前设置下 FID 出现 200-300 左右时，不能直接和论文中 20 左右的数值比较，主要原因是样本量更少，且比较对象不同。
- CS 评测会下载 `openai/clip-vit-large-patch14`，这是用于文图相似度评测的 CLIP 权重，不属于 FLUX 主模型权重。


