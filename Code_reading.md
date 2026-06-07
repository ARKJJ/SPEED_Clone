### Understanding of Code

## `train_erase_null.py`

### 整体作用

这份代码不是传统意义上的“训练”，而是根据 `target / anchor / retain` 的文本语义，直接编辑 Stable Diffusion 的部分 UNet 权重，并保存成一个 `.pt` 文件。

### `get_token_id`

将文本转换为 token id 形式，并返回 `input_ids` 或完整 tokenizer 输出。

关键作用：

1. 把字符串 prompt 变成模型能处理的 token。
2. 支持单个文本，也支持一批文本。
3. 后续 target、anchor、retain 都要先经过这一步。

### `generate_perturbed_embs()`

DPA 模块功能：对 retain embedding 施加定向噪声扰动，扩充 retain 语义样本。

关键步骤：

1. 用 `noise @ P` 把随机噪声限制到某个特定子空间。
2. `perturbed_embs = mini_ret_embs + noise @ P` 得到扰动后的 embedding。
3. `torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1)` 计算扰动样本在当前擦除方向上的响应强度。
4. 只保留响应高于平均值的扰动样本，用来增强 retain 集的代表性。

### `edit_model()`

根据 target、anchor、retain 的文本 embedding 和 SPEED 的矩阵公式，直接计算并修改 UNet 的部分权重。

关键步骤：

1. 选择要编辑哪些权重层，由 `args.params` 决定修改 `KV`、`K` 或 `V`。
2. IEC：对空 prompt 的 embedding 做编码，再对除第一个 token 外的 embedding 做 kmeans 聚类，将第一个 token embedding 和三个聚类中心拼成 `K2`，用于约束编辑过程不要偏移过大。
3. 构造 target 和 anchor 的矩阵：
   - 将目标概念和锚点概念分别编码。
   - 提取关键 token 的 embedding。
   - 组成 target-target 和 anchor-target 的统计矩阵。
4. 构造 retain 集 embedding：
   - 从 csv 中读取保留概念文本。
   - 去掉和 target 重复的项。
   - 编码成 retain 的文本向量。
5. 对每一层计算更新量：
   - 先计算初始擦除方向 `erase_weight`。
   - 然后 SVD 分解当前层权重，得到最小奇异值对应方向，构造投影矩阵 `P0_min`，用于 DPA 扰动。
   - 接着对 retain 统计矩阵 `sum_ret_ret` 做 SVD，奇异值小于阈值的方向组成保留子空间投影矩阵 `P`。
   - 最后按 SPEED 公式解出 `delta_weight`。
6. 用 `layer_weight + delta_weight` 得到编辑后的权重，并写回 `edit_dict`。

### 主程序入口

关键步骤：

1. 定义基础参数、擦除参数和超参数。
2. 解析 `target_concepts` 和 `anchor_concepts`，统一成列表格式，并构造输出文件名。
3. 构造 `retain_texts`：
   - 从 `retain_path` 指定的 csv 中读取。
   - 用 `heads` 指定读取哪一列。
4. 加载 Stable Diffusion v1.4：
   - `tokenizer`
   - `text_encoder`
   - `unet`
5. 调用 `edit_model()` 执行模型编辑。
6. 将编辑后的权重保存为 `.pt` 文件，默认保存在 `logs/checkpoints/` 下。

## `sample.py`

### 文件作用

这份代码负责对原始 Stable Diffusion 和编辑后的 UNet 进行采样，并把生成结果保存下来用于对比。

### `diffusion()`

扩散去噪过程，把随机噪声 latent 一步步变成最终图像 latent。

关键步骤：
1. latent 迭代去噪：从随机噪声开始，在每个时间步逐步减少噪声。
2. 时间步机制：不同时间步对应不同噪声强度，前期决定整体结构，后期补充细节。
3. CFG 机制：一次前向同时计算无条件和有条件预测，再组合得到更强的文本控制效果。

### 参数解析模块

读取命令行参数，控制采样方式、模型路径、输出目录以及测试概念。

关键步骤：
1. 读取基础配置，如 `save_root`、`sd_ckpt`、`seed`。
2. 读取采样配置，如 `mode`、`guidance_scale`、`total_timesteps`、`num_samples`、`batch_size`。
3. 读取擦除任务配置，如 `erase_type`、`target_concept`、`contents`、`edit_ckpt`。

### 概念筛选模块

判断哪些 concept 需要重新采样，避免重复生成已经完整保存的结果。

关键步骤：
1. 将 `contents` 拆成 concept 列表。
2. 如果包含 `edit` 模式，就检查对应输出目录中的图片数量是否完整。
3. 只有未采完的 concept 才会加入待处理列表。

### 模型准备模块

加载原始 Stable Diffusion，以及可选的编辑后 UNet。

关键步骤：
1. 原始模型与编辑模型并存：两者共用同一套 `tokenizer`、`text_encoder` 和 `vae`。
2. 只替换 UNet 权重：说明编辑结果主要体现在 UNet 上，而不是整条 pipeline 都变化。
3. 关闭 `safety checker`：避免外部安全过滤干扰实验评估，尤其是在 `nudity` 任务中。

### Prompt 构造模块

把概念变成真正送给模型的文本条件。

关键步骤：
1. 模板化测试：不是只测一句 prompt，而是用一组不同表达方式覆盖同一概念。
2. 无条件 embedding：构造空 prompt 的 embedding，作为 CFG 的参照分支。
3. 条件 embedding：把当前 prompt 编码成文本向量，作为真正的文本条件输入。

### 采样主循环模块

逐批次、逐 concept、逐 prompt 调用 `diffusion()` 生成 latent 结果。

关键步骤：
1. 同一 latent 对比机制：原始模型和编辑模型使用同一份初始噪声，便于公平比较。
2. CFG 双分支输入机制：把 latent 和 text embedding 都复制成无条件、有条件两路输入。
3. 批量采样机制：一次处理 `batch_size` 张图，提高采样效率。

### 解码与保存模块

把 latent 变成真正图片，并按目录结构保存下来。

关键步骤：
1. VAE 解码机制：将潜空间中的 latent 还原为像素空间图像。
2. 图像后处理：把模型输出从张量格式转成正常可保存的 PIL 图片。
3. 结果组织：分别保存到 `original`、`edit` 和 `combine` 目录中，方便对比。
