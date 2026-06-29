# Code Structure

## FLux 目录结构

```text
FLux/
├── CE_Flux.py
├── sample.py
├── data/
│   ├── instance.csv
│   ├── style.csv
│   ├── 10_celebrity.csv
│   ├── 50_celebrity.csv
│   ├── 100_celebrity.csv
│   ├── i2p_benchmark.csv
│   ├── mscoco.csv
│   └── pretrain/pretrain_sample.sh
├── scripts/
│   ├── eval_few.sh
│   ├── eval_multi.sh
│   └── eval_nudity.sh
└── task-knowledge/
    ├── project-background.md
    ├── environment-and-paths.md
    ├── code-structure.md
    ├── current-task.md
    ├── daily-log.md
    └── finished-task/
```

这个结构说明当前 FLux 目录已经包含代码、数据、脚本和知识库。但要谨慎：并不是目录里所有文件都已经适配 FLUX。当前最可靠的 FLUX 核心代码是 `CE_Flux.py`；`sample.py` 和 `scripts/` 仍明显带有 Stable Diffusion/UNet 流程痕迹，是否可用于 FLUX 待确认。

## `CE_Flux.py`

`CE_Flux.py` 是当前项目最核心的文件，负责把 MEMIT 式层差偏移机制放入 FLUX transformer 中。

主要职责：

- 加载 FLUX pipeline。
- 选择要编辑的 transformer block。
- 在选定 block 中定位 `.attn.add_q_proj` 和 `.attn.add_k_proj`。
- 对 `target_concepts`、`anchor_concepts`、`retain_concepts` 运行 trace。
- 根据 anchor 输出和 target 当前输出之间的差异构造 residual。
- 暂不加入 retain Gram 正则，先只保留 target 到 anchor 的层差偏移主路径。
- 求解闭式权重偏移 `delta`。
- 保存编辑后的局部权重为 `.safetensors`。

关键对象：

| 名称 | 作用 | 为什么重要 |
| --- | --- | --- |
| `MemitFluxConfig` | 保存层范围、trace 步数、正则项、残差缩放等配置 | 这些超参数直接决定编辑强度和稳定性 |
| `get_token_id` | 使用 FLUX tokenizer 将概念文本转成 token ids | token 选择错误会导致编辑目标偏离 |
| `_select_text_qk_modules` | 选择 FLUX transformer 中的文本侧 q/k 投影层 | 决定 MEMIT 偏移实际写入哪里 |
| `_trace_prompt` | 注册 forward hook，记录模块输入和输出 | MEMIT 式层差偏移依赖这些中间激活 |
| `_closed_form_update` | 根据 keys、residuals 和 update 正则求解 delta | 是当前编辑机制的数学核心 |
| `edit_model` | 组织 trace、约束构造和逐层更新 | 是从概念输入到权重更新的主流程 |
| `apply_memit_flux` | 加载模型、调用编辑、保存结果 | 是命令行入口背后的封装函数 |
| `UCE_double_proxy` | `apply_memit_flux` 的别名 | 可能用于兼容旧接口，当前用途待确认 |

## `sample.py`

`sample.py` 当前仍是 Stable Diffusion 风格采样脚本：它加载 `DiffusionPipeline` 后使用 `unet`、`tokenizer`、`text_encoder`、`vae`，并通过 `unet_edit.load_state_dict(torch.load(...))` 加载 `.pt` 权重。

这点很重要，因为它和 `CE_Flux.py` 输出不匹配：

- `CE_Flux.py` 输出 `.safetensors`。
- `sample.py` 期望 `.pt`。
- `CE_Flux.py` 编辑 FLUX transformer q/k。
- `sample.py` 加载并替换 UNet 权重。

因此，目前不能把 `sample.py` 当作 FLUX 版验证脚本。它要么是历史拷贝，要么需要后续改造。

## `data/`

`data/` 包含从 SPEED 任务继承来的概念列表和评估 prompts：

| 文件 | 当前用途 |
| --- | --- |
| `instance.csv` | instance 类概念列表，可作为 retain/target 候选来源 |
| `style.csv` | 艺术风格概念列表 |
| `10_celebrity.csv` | 10 名人擦除/保留评估 prompts |
| `50_celebrity.csv` | 50 名人擦除/保留评估 prompts |
| `100_celebrity.csv` | 100 名人擦除/保留评估 prompts |
| `i2p_benchmark.csv` | I2P 不安全 prompt benchmark |
| `mscoco.csv` | 通用 COCO 文本 prompts |
| `pretrain/pretrain_sample.sh` | 预训练模型 baseline 采样脚本，是否适配 FLUX 待确认 |

这些数据重要，是因为 FLUX 实验最终也需要可重复的 target、retain 和 evaluation prompts。但当前 `CE_Flux.py` 尚未直接读取 CSV，所以数据只是可用资源，不是已经接入的 pipeline。

## `scripts/`

`scripts/` 下包含：

- `eval_few.sh`
- `eval_multi.sh`
- `eval_nudity.sh`

这些脚本目前仍引用 `train_erase_null.py`、`sample2.py`、`src/clip_score_cal.py` 等 FLux 目录中未看到的文件。因此它们更像从 SPEED 主项目复制来的实验脚本，而不是已经完成的 FLUX 实验脚本。

对新人来说，这个判断非常重要：不要直接运行这些脚本并假设它们会评估 `CE_Flux.py`。在运行前必须先确认它们是否已迁移到 FLUX 权重保存和加载方式。

## `task-knowledge/`

该目录是项目知识库，用来记录：

- FLUX 项目背景。
- 运行环境与路径。
- 目录结构和关键文件。
- 当前任务。
- 每日记录。

这个目录不参与模型运行，但对研究协作很重要。它能减少“代码能跑但没人知道为什么这样改”的问题，也能把待确认的研究假设显式记录下来。

## 待确认

- `sample.py` 是否应删除、保留为旧参考，或改造成 FLUX 采样脚本。
- `scripts/` 是否应整体重写为 FLUX 版评估脚本。
- 是否需要新增 `src/` 目录来放 FLUX 采样、token 检查和评估工具。
