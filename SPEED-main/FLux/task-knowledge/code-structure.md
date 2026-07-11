# Code Structure

## 当前目录结构

```text
FLux/
├── CE_Flux.py
├── sample.py
├── data/
│   ├── instance.csv
│   ├── instance_small.csv
│   ├── style.csv
│   ├── style_100.csv
│   ├── 10_celebrity.csv
│   ├── 50_celebrity.csv
│   ├── 100_celebrity.csv
│   ├── i2p_benchmark.csv
│   ├── mscoco.csv
│   └── pretrain/pretrain_sample.sh
└── task-knowledge/
    ├── project-background.md
    ├── code-structure.md
    ├── current-task.md
    ├── daily-log.md
    └── finished-task/
```


当前工作区里 `scripts/` 和若干旧知识文件已不在目录中，因此不要再把它们当作 FLUX 当前流程的一部分。

## CE_Flux.py

`CE_Flux.py` 是当前 FLUX 概念编辑的核心脚本，已经清除了逐行说明注释。它负责加载 FLUX pipeline、追踪 transformer 中的文本侧 attention 投影层，并把闭式解得到的局部权重写入 `.safetensors`。

当前配置入口：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--sd_ckpt` | `black-forest-labs/FLUX.1-schnell` | 基础 FLUX 模型 |
| `--save_path` | `logs/checkpoints` | 编辑权重保存目录 |
| `--file_name` | 时间戳加目标概念后缀 | 输出文件名，不含扩展名 |
| `--seed` | `0` | 目前仅解析，核心 trace 使用 `trace_seed` |
| `--device` | `cuda` | pipeline 和模块运行设备 |
| `--target_concepts` | 必填 | 逗号分隔的目标概念 |
| `--anchor_concepts` | 必填 | 逗号分隔的 anchor 概念；单个 anchor 会复制给所有 target |
| `--retain_path` | `None` | retain CSV 路径 |
| `--heads` | `None` | retain CSV 中要读取的列名；使用 `retain_path` 时必填 |
| `--params` | `KV` | 可选 `Q`、`K`、`V`、`QK`、`KV`、`QKV` |
| `--threshold` | `1e-1` | retain 输入协方差 SVD 的零空间阈值 |
| `--trace_num_steps` | `4` | trace 前向生成步数 |
| `--trace_seed` | `0` | trace 使用的随机种子 |
| `--trace_resolution` | `512` | trace 图像宽高 |
| `--update_lambda` | `1e-4` | 闭式解中的岭正则强度 |
| `--residual_scale` | `1.0` | residual 分摊前的缩放系数 |

关键实现点：

- 默认设置 `HF_ENDPOINT=https://hf-mirror.com`。
- 文本最大长度按模型名自动选择：`schnell` 使用 `256`，其他 FLUX 模型使用 `512`。
- `_select_text_attention_modules` 会在 `transformer_blocks.*` 下选择名称包含 `.attn.add_q_proj`、`.attn.add_k_proj`、`.attn.add_v_proj` 的模块。
- 当前没有 `layer_start`、`layer_end`、`layer_stride` 参数，模块选择覆盖所有匹配的 `transformer_blocks`。
- token 选择使用 `tokenizer_2` 的 `attention_mask`，排除末尾特殊 token；空字符串 anchor/retain 会使用 token 位置 `[0]`。
- retain 文本会先过滤掉包含目标概念完整词匹配的文本。
- `_closed_form_update` 使用 retain 输入协方差的 SVD 构造近似零空间投影，再在投影空间里求解 `delta`。
- 更新方式是 `module.weight = module.weight + delta`，输出只包含被编辑的 transformer 局部权重。

## sample.py

`sample.py` 当前处于未合并状态，文件中仍有 `<<<<<<< HEAD`、`=======`、`>>>>>>>` 冲突标记，因此不能直接运行。

从两个冲突分支可以看出它的目标用途是：

- 加载 FLUX pipeline，并开启 VAE slicing/tiling。
- 支持 `original`、`edit` 或 `original,edit` 采样模式。
- 从 `edit_ckpt` 读取 `CE_Flux.py` 保存的 `.safetensors`，按同名 key 覆盖 `pipe_edit.transformer.state_dict()`。
- 使用 `template_dict[erase_type]` 或 `--prompts` 构造 prompt。
- 按 `seed`、`num_samples`、`batch_size` 生成并保存图片。
- 当同时采样 original/edit 时，额外保存横向拼接图。

当前可见参数配置：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--save_root` | 空字符串 | 结果保存根目录 |
| `--sd_ckpt` | `black-forest-labs/FLUX.1-schnell` | 默认模型 |
| `--model_id` | `None` | 提供时覆盖 `sd_ckpt` |
| `--seed` | `0` | 起始 seed |
| `--device` | `cuda:0` | 运行设备 |
| `--torch_dtype` | `bfloat16` | 可选 `float16`、`bfloat16`、`float32` |
| `--mode` | `original` | 采样模式，逗号分隔 |
| `--guidance_scale` | `0.0` | FLUX schnell 常用 0 |
| `--total_timesteps` | `4` | 采样步数 |
| `--num_samples` | `10` | 每个 prompt 的样本数 |
| `--batch_size` | `10` | 每轮 seed 数量 |
| `--prompts` | `None` | 分号分隔的自定义 prompt 模板 |
| `--height` | `512` | 输出高度 |
| `--width` | `512` | 输出宽度 |
| `--max_sequence_length` | `None` | 未指定时按模型类型自动设置 |
| `--erase_type` | 空字符串 | 模板类别，如 `instance`、`style`、`celebrity` |
| `--target_concept` | 空字符串 | 保存路径中的目标概念名 |
| `--contents` | 空字符串 | 逗号分隔的待采样概念 |
| `--edit_ckpt` | `None` | 编辑权重文件路径 |

注意：当前 `sample.py` 中没有可见的 `--strict_edit_load` 参数，旧命令里的该参数需要删除或等代码补齐后再使用。

## data/

`data/` 主要保存 target、retain 和 evaluation prompt 资源。当前 `CE_Flux.py` 只直接使用 `retain_path` 和 `heads` 读取 CSV，其他 CSV 是否用于评估取决于外部命令和 `sample.py` 后续修复。

| 文件 | 当前用途 |
| --- | --- |
| `instance.csv` | instance 类概念候选 |
| `instance_small.csv` | 小规模 instance retain/测试列表 |
| `style.csv` | 风格概念候选 |
| `style_100.csv` | 小规模 style retain/测试列表 |
| `10_celebrity.csv` | celebrity 评估或实验概念 |
| `50_celebrity.csv` | celebrity 评估或实验概念 |
| `100_celebrity.csv` | celebrity 评估或实验概念 |
| `i2p_benchmark.csv` | I2P prompt benchmark |
| `mscoco.csv` | 通用 COCO prompt 文本 |
| `pretrain/pretrain_sample.sh` | 旧采样脚本，是否适配当前 FLUX 流程未确认 |


