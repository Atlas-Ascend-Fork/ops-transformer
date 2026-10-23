# SplitQkvRmsnormMrope 测试

`ut/op_host` 覆盖 Gate 开启/关闭时的输出 Shape，并验证非法 rank、head 数和 QKV 宽度会被拒绝。运行：

```bash
bash build.sh --ophost_test --experimental --soc=ascend910b \
    --ops=split_qkv_rmsnorm_mrope
```

端到端 ACLNN 用例位于 `../examples/test_aclnn_split_qkv_rmsnorm_mrope.cpp`，使用非零 Q/K、weight、
cos/sin 和 CPU 参考结果检查 RMSNorm、仿射、MRoPE 及 Q/K 输出；同时覆盖无 bias 的单 token 优化路径、
带 Q/K bias 的多 token 路径、非零 Gate/V BF16 原始位直通，并验证非法 `eps` 和 head 数会被拒绝。
正式验收矩阵和性能基线由验收交付件保存，不复制进开源仓目录。
