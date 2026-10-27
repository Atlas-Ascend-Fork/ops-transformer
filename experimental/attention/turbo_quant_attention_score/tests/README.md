# TurboQuantAttentionScore 测试

`ut/op_host` 覆盖两个内部节点的 InferShape 和 InferDataType，并包含 `Nkv > 16` 时 metadata
按 32 字节 UB 行搬运的布局回归检查。运行：

```bash
bash build.sh --ophost_test --experimental --soc=ascend910b \
    --ops=turbo_quant_attention_score
```

端到端 ACLNN 用例位于 `../examples/test_aclnn_turbo_quant_attention_score.cpp`，使用非零 Query、量化
索引、QJL 符号、norm 和 gamma 与 CPU 参考实现比对，覆盖 2/3/4-bit 的 Decode、Prefill 数值路径，
并验证序列尾部及非法物理页屏蔽。正式验收自验矩阵为 Decode 4K/64K/128K、Prefill 4K/32K，使用
任务交付件中的确定性数据生成器和固定 baseline 执行。
