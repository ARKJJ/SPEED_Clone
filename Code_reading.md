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
