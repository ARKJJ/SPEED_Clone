### Understanding of Code

## `train_erase_null.py`

### `get_token_id`

将文本转换为 token id 形式。

### `generate_perturbed_embs()`

DPA 模块功能：对 retain 施加定向噪声，增强语义。

关键步骤：
1. 用 `noise @ P` 把噪声限制到某个特定子空间。
2. `perturbed_embs = mini_ret_embs + noise @ P` 得到扰动后的 embedding。
3. `torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1)` 计算扰动样本在当前擦除方向上的响应强度，只保留响应强于平均值的样本。

### `edit_model()`

根据文本 embedding 和矩阵公式编辑权重部分。

关键步骤：
1. 选择要编辑哪些权重层，由 `args.params` 决定修改 `KV`、`K`、`V`。
2. IEC：对除第一个 token 外的 embedding 做 kmeans 聚类，将第一个 token embedding 和三个聚类中心拼成 `K2`，用于约束擦除过程不要偏移。
3. 构造 target 和 anchor 的矩阵。
4. 构造 retain 集 embedding。
5. 对每一层计算擦除更新量：
   - 首先计算初始方向。
   - 然后 SVD 分解权重，得到当前层权重最小奇异值对应的方向，构造投影矩阵 `P0`，这个方向会用于 DPA。
   - 接着求保留子空间，对 `sum_ret_ret` 做 SVD，奇异值小于阈值的方向组成投影矩阵 `P`。
   - 最后解出 `delta_weight`。

### 主程序入口

关键步骤：

1. 定义基础参数等。
2. 解析 target 和 anchor，控制输入和输出文件格式。
3. 构造 `retain_texts`，从 csv 中读取，并加载扩散模型 SD1.4。
4. 执行模型编辑。

## 'sample.py'

### 'diffusion()'

扩散去噪过程，把随机噪声 latent 一步步变成最终图像 latent

关键步骤：
1.latent 迭代去噪：从噪声开始，逐步生成图像。
2.时间步机制：每一步噪声强度不同，越往后越精细。
3.CFG 机制：一次前向同时算无条件和有条件预测，再组合得到更强的文本控制。

### '模型准备模块'

加载原始 Stable Diffusion，以及可选的编辑后 UNet。

关键步骤：
1.原始模型与编辑模型并存：两者共用同一套 tokenizer、text encoder、vae。
2.只替换 UNet 权重：说明编辑结果主要体现在 UNet 上。
3.关闭 safety checker：避免外部安全过滤干扰实验评估，尤其是 nudity 类任务。

### 'Prompt 构造模块'

把概念变成真正送给模型的文本条件。

关键步骤：
1.模板化测试：不是只测一句 prompt，而是用一组不同表达方式覆盖同一概念。
2.无条件 embedding：为 CFG 提供“参照分支”。

### '采样主循环模块'

逐批次、逐 concept、逐 prompt 调用 diffusion() 生成 latent 结果。

关键步骤：
1.同一 latent 对比机制：原始模型和编辑模型使用同一份初始噪声。
2.CFG双分支输入机制

解码与保存模块

把 latent 变成真正图片，并按目录结构保存下来.

关键机制：
1.VAE 解码机制。