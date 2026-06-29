# Environment and Paths

## 工作目录

当前 FLUX 项目的工作目录是：

```bash
cd /Users/ark/SPEED_Clone/SPEED-main/FLux
```

后续命令和相对路径应优先以这个目录为基准。这个约定重要，因为 `CE_Flux.py` 默认将编辑权重保存到当前目录下的 `./models`，而 `data/`、`scripts/` 也都位于 `FLux` 内部。如果工作目录错了，输出文件和数据读取位置都会偏移。

## 核心环境

`CE_Flux.py` 依赖的核心包包括：

- `torch`
- `diffusers`
- `safetensors`

其中 `diffusers` 必须支持 FLUX pipeline，`safetensors` 用于保存编辑后的局部权重。运行 FLUX 还需要 GPU 环境；脚本默认设备为 `cuda:0`，默认权重精度为 `torch.bfloat16`。

这点重要，是因为当前任务不是轻量 CPU 脚本。模型加载、trace 生成、forward hook 记录中间激活都会占用显存。环境不匹配时，错误往往出现在模型加载或 CUDA 执行阶段，而不是出现在编辑公式本身。

## 模型路径

默认 FLUX 模型：

```text
black-forest-labs/FLUX.1-schnell
```

对应命令参数：

```bash
--model_id "black-forest-labs/FLUX.1-schnell"
```

`CE_Flux.py` 会根据模型名中是否包含 `schnell` 设置 `max_sequence_length`：`schnell` 默认 256，否则默认 512。这个细节重要，因为 token 长度会影响 `_content_token_indices` 选择哪些 token 参与 trace。

## 代码路径

```text
FLux/
├── CE_Flux.py
├── sample.py
├── data/
├── scripts/
└── task-knowledge/
```

当前真正和 FLUX 编辑直接相关的是 `CE_Flux.py`。`sample.py` 和 `scripts/` 当前仍保留 Stable Diffusion 风格逻辑，是否已适配 FLUX 待确认，不能直接当作 FLUX 评估闭环使用。

## 数据路径

FLUX 目录下已有数据：

```text
FLux/data/
├── instance.csv
├── style.csv
├── 10_celebrity.csv
├── 50_celebrity.csv
├── 100_celebrity.csv
├── i2p_benchmark.csv
├── mscoco.csv
└── pretrain/pretrain_sample.sh
```

这些数据重要，是因为后续可以复用 SPEED 的概念列表、保留概念和评估 prompts。但当前 `CE_Flux.py` 的命令行接口直接接收 `target_concepts`、`anchor_concepts`、`retain_concepts`，并没有直接读取这些 CSV。因此，数据文件已经在目录中，并不等于 FLUX 版本已经自动使用它们。

## 输出路径

`CE_Flux.py` 默认输出目录：

```text
FLux/models/
```

默认输出文件：

```text
FLux/models/flux_memit_qk_test.safetensors
```

可以通过以下参数修改：

```bash
--save_dir "./models"
--exp_name "your_experiment_name"
```

最终文件名为：

```text
{save_dir}/{exp_name}.safetensors
```

这个输出文件只保存被编辑的 q/k 模块权重，例如 `transformer_blocks.6.attn.add_q_proj.weight`。它不是完整模型，也不是原始 SPEED 的 `.pt` UNet checkpoint。后续采样时必须写对应的 FLUX 加载逻辑。

## 常用命令

查看参数：

```bash
python3 CE_Flux.py --help
```

对象概念示例：

```bash
python3 CE_Flux.py \
  --concept_type object \
  --target_concepts "Snoopy" \
  --anchor_concepts "" \
  --retain_concepts "Mickey;Hello Kitty;Pikachu" \
  --save_dir "./models" \
  --exp_name "snoopy_flux_qk"
```

艺术风格示例：

```bash
python3 CE_Flux.py \
  --concept_type art \
  --target_concepts "Van Gogh" \
  --anchor_concepts "art" \
  --retain_concepts "Picasso;Monet;Paul Gauguin" \
  --expand_prompts true \
  --save_dir "./models" \
  --exp_name "vangogh_flux_qk"
```

快速 smoke test 可以减少 trace 步数和编辑层数：

```bash
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

## 参数注意事项

- 多个概念使用分号 `;` 分隔，不是逗号。
- 对外统一使用 SPEED 风格命名：`--target_concepts`、`--anchor_concepts`、`--retain_concepts`。
- 旧版参数名仅作为隐藏兼容别名保留，不建议继续写入新命令或文档。
- `--anchor_concepts` 为空时，对 object 类型默认不会自动给出语义 anchor；代码会进一步依赖 `--retain_concepts` 作为 anchor 来源。
- 当前版本暂不使用 `retain_concepts` 构建 retain Gram 正则。
- `--replace_indices` 默认是 `all`，表示使用目标概念的全部有效 token。若手动指定 token index，必须先核对 tokenizer 输出。
- `--layer_stride` 必须大于 0，`--layer_end` 必须大于 `--layer_start`。

## 待确认

- 当前机器是否具备 FLUX 模型访问权限。
- `sample.py` 是否计划改造成 FLUX 采样脚本，还是只作为从 SPEED 拷贝来的旧文件保留。
- `scripts/` 中脚本是否仍然有效；当前看它们引用了一些 FLux 目录下不存在的文件。
- 后续是否统一把 FLUX 输出写到 `models/`，还是按实验类型拆分到 `logs/`。
