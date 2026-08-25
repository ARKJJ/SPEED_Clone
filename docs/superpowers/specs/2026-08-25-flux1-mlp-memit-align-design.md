# Flux1 MLP-MEMIT 对齐 Flux2 的重构设计

## 目标

让 `SPEED-main/FLux/Flux1/mlp_memit.py` 的代码组织、变量风格和主流程尽量接近当前 `Flux2/mlp_memit.py`，删除不必要的冗余代码，同时保持 Flux1 的模型适配和 MEMIT 编辑语义。

## 保留范围

- Flux1 使用 `DiffusionPipeline`。
- Flux1 使用 `pipeline.tokenizer_2` 的 T5 tokenization。
- Flux1 的 MLP 模块后缀保持 `.ff_context.net.2`。
- 保留当前 MEMIT 的最终 MLP 输出 residual、`remaining_counts` 层间分配、retain projector 和闭式 solve。
- 保留 `residual_scale` 参数及其对 residual 的缩放作用。
- 保留 Flux1 当前的序列长度默认值 `256`。

## 删除或压缩范围

- 删除 `try/finally` hook 清理包装，使 tracing 结构与 Flux2 对齐。
- 删除额外的 CLI 参数校验和冗长错误包装，采用 Flux2 的直接解析风格。
- 合并只被主流程调用一次、且没有独立复用价值的中间变量或过度展开的表达式。
- 将 tracing、模块筛选、retain 收集和主入口的格式压缩为 Flux2 的组织方式。

## 不改变的行为

- 不把 Flux1 的 tokenizer 路径改成 Flux2 的 chat template。
- 不把 Flux1 的模块名改成 Flux2 的 `.ff_context.linear_out`。
- 不移除 `residual_scale`。
- 不改变目标、anchor、retain 的 tracing 顺序和闭式更新的矩阵计算。
- 不修改 Flux2 文件。

## 验证

1. 添加或更新静态测试，锁定 Flux1 的 pipeline 类型、tokenizer、MLP 后缀和 `residual_scale` 路径。
2. 先验证测试在未重构版本上能正确捕获目标结构。
3. 完成重构后运行静态测试、Python 编译检查和 `git diff --check`。
4. 不把静态检查当作 GPU 模型加载或图像质量验证。
