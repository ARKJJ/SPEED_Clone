# Code Pipeline Understanding: SPEED Model Editing and Sampling

## 1. 整体任务目标

这个仓库的核心不是重新训练一个完整的 Stable Diffusion，而是把“概念擦除”理解成一次直接的权重编辑任务。`train_erase_null.py` 读取 `target / anchor / retain` 文本语义，直接修改 Stable Diffusion UNet 中 cross-attention 的部分权重，输出一个只包含编辑后权重的 `.pt` 文件；`sample.py` 再加载这个 edited checkpoint，在相同 prompt 和相同初始 latent 条件下，对比原始模型和编辑模型的生成结果。

换句话说，整条流程可以分成两个阶段：

1. 编辑阶段：把文本语义约束转成 UNet 权重更新。
2. 验证阶段：加载编辑后的 UNet，与原始模型做公平采样对比。

## 2. 模块化代码理解

### 模块 1：参数与任务配置模块

**模块作用**

这是整个 pipeline 的入口。它负责把用户从命令行传入的擦除任务、保留集、采样模式和保存路径，整理成后续模块可直接使用的标准化配置。

**对应代码**

- `train_erase_null.py`
  - `argparse` 参数定义
  - `target_concepts = [con.strip() for con in args.target_concepts.split(',')]`
  - `anchor_concepts = [x.strip() for x in anchor_concepts.split(',')]`
  - `retain_path`, `heads`, `params`, `aug_num`, `threshold`, `retain_scale`, `lamb`
  - `file_suffix`、`save_path`、`file_name` 的构造
- `sample.py`
  - `argparse` 参数定义
  - `mode_list = args.mode.replace(' ', '').split(',')`
  - `concept_list_tmp = [item.strip() for item in args.contents.split(',')]`
  - `edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])`

**输入输出**

- 输入
  - 用户命令行传入的 `target_concepts`、`anchor_concepts`、`retain_path`、`heads`
  - 编辑超参数 `params / aug_num / threshold / retain_scale / lamb`
  - 采样参数 `mode / total_timesteps / guidance_scale / num_samples / batch_size`
  - 模型路径 `sd_ckpt` 和编辑权重路径 `edit_ckpt`
- 输出
  - 标准化后的 `target_concepts` 列表
  - 与之对齐的 `anchor_concepts` 列表
  - `retain_texts`
  - 要编辑的权重类型设置 `args.params`
  - checkpoint 保存路径和采样输出路径

**与其他模块的关系**

这个模块决定后面所有模块“读什么、改什么、存到哪里”。尤其重要的是：

- `args.params` 决定 UNet 权重更新模块到底编辑 `K`、`V` 还是 `KV`。
- `retain_path + heads` 决定 retain 文本从哪个 CSV、哪几列读取。
- `edit_ckpt` 把 `train_erase_null.py` 的输出连接到 `sample.py` 的输入。

### 模块 2：文本编码模块

**模块作用**

这个模块把自然语言 prompt 转成 Stable Diffusion 可以使用的 token ids 和 text embeddings。编辑阶段和采样阶段都依赖这一步，因为概念擦除本质上是“根据文本语义来改权重”，采样对比本质上是“根据文本语义来生成图像”。

**对应代码**

- `train_erase_null.py`
  - `get_token_id(prompt, tokenizer=None, return_ids_only=True)`
  - `pipeline.tokenizer(...)`
  - `pipeline.text_encoder(...).last_hidden_state`
  - target / anchor / retain / empty prompt 的编码过程都在 `edit_model()` 中
- `sample.py`
  - `src.utils.get_token`
  - `src.utils.get_textencoding`
  - `uncond_embedding = get_textencoding(get_token('', tokenizer), text_encoder)`
  - `embedding = get_textencoding(get_token(prompt, tokenizer), text_encoder)`

**输入输出**

- 输入
  - `target_concepts`
  - `anchor_concepts`
  - `retain_texts`
  - 空 prompt `''`
  - 采样 prompt
- 输出
  - target embedding
  - anchor embedding
  - retain embedding
  - unconditional embedding
  - sampling conditional embedding

**与其他模块的关系**

- target / anchor / retain / empty prompt embedding 会进入编辑约束构造模块。
- unconditional embedding 和 sampling prompt embedding 会进入采样对比模块中的 CFG 双分支。

**代码实现上的关键点**

1. `get_token_id()` 和 `get_token()` 先把字符串 prompt 转成 token ids。
2. `text_encoder(...)` 再把 token ids 转成 token 级别的隐向量表示。
3. `train_erase_null.py` 对不同文本的取法不完全一样：
   - `target` 和 `anchor` 通常取最后一个有效 token 的 embedding，作为概念代表向量。
   - `nudity` 任务特判为使用全部非特殊 token。
   - `retain` 文本默认也取最后一个有效 token。
   - 空 prompt 用全部 token embedding，再进一步做 IEC 约束。

### 模块 3：编辑约束构造模块

**模块作用**

这是 `train_erase_null.py` 的核心前半部分。它负责把“删掉 target”“靠近 anchor”“尽量保留 retain”“不要让空语义漂移太大”这些自然语言目标，变成后面可直接作用于 UNet 权重的矩阵约束。

**对应代码**

- `train_erase_null.py`
  - `edit_model(...)`
  - IEC 相关代码：
    - `null_inputs = get_token_id('', ...)`
    - `null_hidden = pipeline.text_encoder(...).last_hidden_state[0]`
    - `cluster_ids, cluster_centers = kmeans(...)`
    - `K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T`
  - target / anchor 统计矩阵：
    - `sum_target_target.append(target_embs.T @ target_embs)`
    - `sum_anchor_target.append(anchor_embs.T @ target_embs)`
  - retain 统计构造前的数据准备：
    - `retain_texts = [text for text in retain_texts if not any(...)]`
    - `last_ret_embs.append(...)`

**输入输出**

- 输入
  - target embedding
  - anchor embedding
  - retain embedding
  - empty prompt embedding
  - 当前层原始权重 `layer_weight`
- 输出
  - `sum_target_target`
  - `sum_anchor_target`
  - retain 侧的 embedding 集合 `last_ret_embs`
  - IEC 约束矩阵 `K2`
  - 后续计算 `delta_weight` 所需的约束项

**与其他模块的关系**

这个模块把文本语义变成矩阵形式，供 DPA 扰动增强模块和 UNet 权重更新模块使用。没有这一步，后面就无法把“语义上的擦除和保留目标”写进线性代数公式。

**为什么这些约束都需要**

1. target / anchor 约束
   - `sum_target_target` 表示目标概念自身的统计方向。
   - `sum_anchor_target` 表示希望把 target 推向 anchor 的方向。
   - 两者的差决定“擦除或重定向”的基本更新方向。
2. retain 约束
   - retain 集告诉模型哪些非目标概念能力要尽量保住。
   - 它后面会被整理成 retain 子空间，用来限制更新不要破坏过多非目标语义。
3. IEC 约束
   - 空 prompt 编码出的 `null_hidden` 被聚成几个代表性方向，组成 `K2`。
   - 其目的不是增强某个目标概念，而是约束编辑过程不要让基础语义空间整体漂移过大。

### 模块 4：DPA 扰动增强模块

**模块作用**

这是 retain 约束的增强模块。它不是直接修改权重，而是先扩展 retain 样本，使“应该被保留的语义”在编辑时更稳定、更有覆盖性。

**对应代码**

- `train_erase_null.py`
  - `generate_perturbed_embs(ret_embs, P, erase_weight, num_per_sample, mini_batch=8)`
  - `noise = torch.randn_like(mini_ret_embs)`
  - `perturbed_embs = mini_ret_embs + noise @ P`
  - `torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1)`
  - `return out_embs[norm_list > norm_list.mean()].unsqueeze(1)`
  - 在 `edit_model()` 中的调用：
    - `P0_min = V0[:, -1:] @ V0[:, -1:].T`
    - `chunk_ret_embs = torch.cat([chunk_ret_embs, generate_perturbed_embs(...)], dim=0)`

**输入输出**

- 输入
  - retain embedding `chunk_ret_embs`
  - 投影矩阵 `P0_min`
  - 当前层的 `erase_weight`
  - 扰动强度和样本数 `args.aug_num`
- 输出
  - 扰动后的 retain embeddings
  - 增强后的 retain 样本集合

**与其他模块的关系**

它服务于 retain 约束构造和权重更新。增强后的 retain 样本会进入 `sum_ret_ret` 的统计计算，最终影响 `P` 和 `delta_weight`。

**为什么要这样做**

1. 为什么扰动 retain embedding
   - 原始 retain 文本数量有限，直接统计可能不够稳。
   - 加扰动可以让 retain 约束覆盖 retain 概念附近的一片局部语义区域。
2. 为什么扰动限制在特定子空间
   - `noise @ P0_min` 不是任意乱加噪声，而是投到当前层权重最弱响应的方向上。
   - 这样做更像是在低响应子空间中做局部扩展，避免完全偏离 retain 语义。
3. 为什么筛选响应强的扰动样本
   - 代码用 `erase_weight` 测试扰动样本的响应强度，只保留高于平均值的样本。
   - 这些样本更可能对“编辑时的保留约束”真正起作用。

### 模块 5：UNet 权重更新模块

**模块作用**

这是 `train_erase_null.py` 的核心后半部分。前面模块只是把语义目标整理成矩阵，这个模块才真正把这些矩阵变成每一层 cross-attention 权重的实际更新量 `delta_weight`。

**对应代码**

- `train_erase_null.py`
  - 选择可编辑层：
    - `if args.params == 'KV'`
    - `elif args.params == 'V'`
    - `elif args.params == 'K'`
  - 遍历每个要编辑的权重层：
    - `for (layer_name, layer_weight) in tqdm(edit_dict.items(), desc="Model Editing")`
  - 基础擦除方向：
    - `erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()`
  - 当前层 SVD：
    - `(U0, S0, V0) = torch.svd(layer_weight)`
    - `P0_min = V0[:, -1:] @ V0[:, -1:].T`
  - retain 统计：
    - `sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))`
    - `sum_ret_ret = torch.stack(...).sum(0) / valid_num`
  - retain 子空间投影：
    - `(U, S, V) = torch.svd(sum_ret_ret)`
    - `P = U[:, S < args.threshold] @ U[:, S < args.threshold].T`
  - 公式求解：
    - `M = (sum_target_target @ P + args.retain_scale * I).inverse()`
    - `delta_weight = ...`
  - 写回结果：
    - `edit_dict[layer_name] = layer_weight + delta_weight`

**输入输出**

- 输入
  - 原始 UNet layer weight
  - `sum_target_target`
  - `sum_anchor_target`
  - `sum_ret_ret`
  - `K2`
  - DPA 增强后的 retain embeddings
  - `threshold / retain_scale / lamb / params`
- 输出
  - `delta_weight`
  - edited layer weight
  - 最终 `edit_dict`

**与其他模块的关系**

这个模块把“文本层面的语义编辑目标”真正转成“模型层面的参数变化”，它的输出随后进入 checkpoint 保存模块。

**几个关键变量怎么理解**

1. `edit_dict`
   - 不是整个 UNet，而是被挑出来准备编辑的那部分权重字典。
   - 按 `args.params` 只包含 cross-attention 的 `to_k`、`to_v` 或二者。
2. `erase_weight`
   - 可以把它理解成当前层对“target 到 anchor 的编辑目标”的基础响应方向。
   - 它既参与初步筛选 retain 样本，也参与最终权重公式。
3. `P`
   - 来自 retain 统计矩阵的 SVD。
   - 它把更新限制在 retain 相关的低奇异值子空间内，用于减少对非目标语义的破坏。
4. `delta_weight`
   - 这是最终真正加到当前层上的增量。
   - 公式里同时融合了 target-anchor 编辑目标、retain 保留目标和 IEC 约束。

### 模块 6：checkpoint 保存与加载模块

**模块作用**

这是连接 `train_erase_null.py` 和 `sample.py` 的桥梁。前半段负责把编辑结果存下来，后半段负责在采样时把这些修改过的 UNet 权重重新装入模型。

**对应代码**

- `train_erase_null.py`
  - `save_path = args.save_path or "logs/checkpoints"`
  - `file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{file_suffix}"`
  - `torch.save(edit_dict, os.path.join(save_path, f"{file_name}.pt"))`
- `sample.py`
  - `unet_edit = copy.deepcopy(unet)`
  - `edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])`
  - `unet_edit.load_state_dict(torch.load(edit_path, map_location='cpu'), strict=False)`

**输入输出**

- 输入
  - `edit_dict`
  - checkpoint 保存路径
  - 采样阶段的 `edit_ckpt`
- 输出
  - `.pt` edited checkpoint
  - 采样阶段加载好的 `unet_edit`

**与其他模块的关系**

- 前面模块解决“怎么改模型”。
- 这一模块负责把编辑结果物化成可复用文件。
- 后面的采样模块再用这个文件验证“改完以后生成会发生什么变化”。

**为什么这里只替换 UNet 权重**

代码里采样阶段并没有重建一套全新的编辑版 diffusion pipeline，而是：

1. 先加载一个原始 `DiffusionPipeline`。
2. 再 `copy.deepcopy(unet)` 得到 `unet_edit`。
3. 只把 `.pt` 中保存的编辑后 UNet 权重载入 `unet_edit`。

这说明 SPEED 的编辑对象主要是 UNet cross-attention 参数，而不是 tokenizer、text encoder、VAE 或 scheduler。

### 模块 7：采样对比模块

**模块作用**

这是 `sample.py` 的核心模块。它负责在相同采样条件下分别调用原始 UNet 和编辑后 UNet，从而得到可以直接比较的生成结果。

**对应代码**

- `sample.py`
  - `diffusion(...)`
  - `pipe = DiffusionPipeline.from_pretrained(...)`
  - `pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)`
  - `prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]`
  - `latent = torch.randn(bs, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)`
  - `text_embeddings=torch.cat([uncond_embedding] * bs + [embedding] * bs, dim=0)`
  - `save_images['original'] = diffusion(...)`
  - `save_images['edit'] = diffusion(...)`
- 依赖的共享辅助函数
  - `src.utils.get_token`
  - `src.utils.get_textencoding`
  - `src.template.template_dict`

**输入输出**

- 输入
  - original Stable Diffusion 的 `unet`
  - edited UNet `unet_edit`
  - sampling prompts
  - unconditional / conditional embeddings
  - 随机初始 latent
  - scheduler 与采样超参数
- 输出
  - original image latent
  - edited image latent

**与其他模块的关系**

它使用 checkpoint 模块提供的 `unet_edit`，再把生成得到的 latent 交给图像解码与保存模块。

**代码里如何保证对比公平**

1. 同一份初始 latent
   - 在每个采样批次里先生成一次 `latent`。
   - 后续 `original` 和 `edit` 都共用这同一份 latent。
2. 同一份 prompt embedding
   - `embedding` 是针对当前 prompt 只算一次。
   - 原始模型和编辑模型都使用同一份条件 embedding。
3. 同一 scheduler 和超参数
   - 二者共用 `pipe.scheduler`、`guidance_scale`、`total_timesteps`。

**`diffusion()` 在做什么**

1. `scheduler.set_timesteps(total_timesteps)` 设定完整去噪时间步。
2. 每一步都把 latent 复制成两份：
   - 一份对应无条件分支
   - 一份对应有条件分支
3. UNet 预测噪声后，用 CFG 公式组合：
   - `noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)`
4. `scheduler.step(...)` 用新的噪声预测更新 latent。

这就是从随机噪声逐步去噪到最终图像 latent 的过程。

### 模块 8：图像解码与结果保存模块

**模块作用**

这是整个 pipeline 的最终输出模块。它把 `diffusion()` 产生的 latent 解码成 RGB 图像，并按目录结构保存为原始图、编辑图和拼接对比图。

**对应代码**

- `sample.py`
  - `vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]`
  - `process_img(...)`
  - `os.makedirs(os.path.join(save_path, mode), exist_ok=True)`
  - `os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)`
  - `decoded_imgs = {...}`
  - `combine_images_horizontally(...)`
  - `decoded_imgs[mode][idx].save(...)`
  - `img_combined.save(...)`
- `src.utils.process_img`
  - 张量到 `[0, 255]` 像素范围
  - `numpy` 转换
  - `Image.fromarray(...)`

**输入输出**

- 输入
  - original / edited image latent
  - VAE
  - 保存路径
- 输出
  - `original` 目录下的原始模型生成图
  - `edit` 目录下的编辑模型生成图
  - `combine` 目录下的横向拼接对比图

**与其他模块的关系**

它接收采样对比模块输出的 latent，产出最终可视化结果，供研究者判断：

- target concept 是否被成功擦除或转向
- retain concept 是否仍然保留
- 编辑副作用是否过大

**为什么要分成 `original / edit / combine`**

1. `original`
   - 保留基线结果，便于看编辑前模型会生成什么。
2. `edit`
   - 单独观察编辑后模型的输出。
3. `combine`
   - 把两张图横向放一起，方便人工快速比较是否发生了目标中的变化。

## 3. 关键数据流总结

### 数据流 1：prompt string → token ids → text embedding

1. 文本先进入 `tokenizer(...)`。
2. 得到固定长度的 token ids。
3. token ids 进入 `text_encoder(...)`。
4. 得到 token 级 embedding。
5. 在编辑阶段，从中取 target / anchor / retain / null 的表示。
6. 在采样阶段，从中取 unconditional / conditional prompt 的表示。

### 数据流 2：target / anchor / retain embedding → 统计矩阵 → `delta_weight`

1. target embedding 构造 `sum_target_target`。
2. anchor 和 target 一起构造 `sum_anchor_target`。
3. retain embedding 经过筛选和 DPA 扰动增强后，构造 `sum_ret_ret`。
4. null prompt embedding 经聚类形成 `K2`。
5. 这些矩阵一起进入 `delta_weight` 的求解公式。

### 数据流 3：original UNet weight + `delta_weight` → edited UNet weight

1. 每个可编辑层先取出 `layer_weight`。
2. 根据 target / anchor / retain / IEC 约束计算 `delta_weight`。
3. 用 `layer_weight + delta_weight` 得到 edited layer weight。
4. 写回 `edit_dict[layer_name]`。

### 数据流 4：edited UNet weight → `.pt` checkpoint → `sample.py`

1. `edit_dict` 被 `torch.save(...)` 存成 `.pt` 文件。
2. `sample.py` 复制原始 `unet` 得到 `unet_edit`。
3. 再通过 `load_state_dict(...)` 把 `.pt` 中的编辑结果加载进去。

### 数据流 5：random latent + text embedding → diffusion denoising → image latent

1. 先采样随机噪声 latent。
2. 将 unconditional 和 conditional embedding 拼接后送进 UNet。
3. `diffusion()` 在多个 timestep 上反复预测噪声并更新 latent。
4. 输出最终图像 latent。

### 数据流 6：image latent → VAE decode → saved image

1. latent 先除以 `vae.config.scaling_factor`。
2. 再送入 `vae.decode(...)`。
3. `process_img(...)` 把结果转成正常 PIL 图像。
4. 最终保存到 `original / edit / combine` 目录。

## 4. `train_erase_null.py` 和 `sample.py` 的衔接关系

这两个文件不是孤立的，而是前后相接的一条实验链路。

`train_erase_null.py` 负责回答的问题是：“怎样根据文本语义直接修改模型？”  
它读取 target、anchor 和 retain 文本，通过文本编码、矩阵约束构造、DPA 增强和 closed-form 权重更新，最终输出一个 edited UNet checkpoint。

`sample.py` 负责回答的问题是：“改完以后效果如何？”  
它加载原始 Stable Diffusion 和 edited UNet，在相同 prompt、相同 latent、相同 scheduler 条件下采样，并保存对比结果。

两者通过 `.pt` checkpoint 串起来：

1. `train_erase_null.py` 产出 `.pt`
2. `sample.py` 读取 `.pt`
3. `sample.py` 用生成结果验证编辑是否成功

所以如果把整个仓库看成一个 pipeline，那么：

- 前者解决“怎么改模型”
- 后者解决“改完后如何验证”

## 5. 汇报总结

这份代码可以理解为一个模型编辑与验证 pipeline。第一阶段通过 `train_erase_null.py` 根据 `target`、`anchor` 和 `retain` 文本语义，直接修改 Stable Diffusion UNet 的部分 cross-attention 权重，并保存 edited checkpoint；第二阶段通过 `sample.py` 加载该 checkpoint，在相同 prompt 和相同 initial latent 下对比 original model 与 edited model 的生成结果，从而评估目标概念是否被有效擦除，同时非目标概念是否被保留。
