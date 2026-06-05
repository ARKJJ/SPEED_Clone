###understanding of code###

##train_erase_null.py##
#get_token_id#
将文本转换为token id形式

#generate_perturbed_embs()#
DPA模块功能：对retain 施加定向噪声，增强语义。
关键步骤：1.用 noise @ P 把噪声限制到某个特定子空间；
        2.perturbed_embs = mini_ret_embs + noise @ P得到扰动后的embedding；
        3.torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1)计算扰动样本在当前擦除方向上的响应强度，只保留响应强于平均值的样本；

#edit_model()#
根据文本embedding和矩阵公式编辑权重部分。
关键步骤：1.选择要编辑哪些权重层，又args.params决定修改KV，K，V；
        2.AIC：对除第一个token外的embedding做kmeans聚类，将第一个token embedding和三个聚类中心拼成K2，用于约束擦除过程不要偏移；
        3.构造target和anchor的矩阵
        4.构造retain集embedding
        5.对每一层计算擦除更新量：首先计算初始方向，然后SVD分解权重得到当前层权重最小奇异值对应的方向，构造投影矩阵P0，这个方向会DPA；接着求保留子空间，对 sum_ret_ret 做 SVD,奇异值小于阈值的方向组成投影矩阵P，最后解出delta_weight

#主程序入口#
关键步骤：1.定义基础参数等
        2.解析target和anchor，控制输入和输出文件格式
        3.构造retain_texts，从csv中读取，并加载扩散模型SD1.4
        4.执行模型编辑