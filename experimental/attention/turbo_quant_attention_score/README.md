# TurboQuantAttentionScore

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR/950DT 系列产品 | × |
| Atlas A3 系列产品 | × |
| Atlas A2 系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`TurboQuantAttentionScore` 使用量化分页 Key cache 计算 attention score。调用方提供经过旋转和 QJL
变换的 Query，以及由索引、QJL 符号、范数和校正系数组成的 2/3/4-bit Key cache。算子解码量化
Key 并计算：

```text
key_hat = Decode(key_cache_idx, key_cache_qjl,
                 key_cache_norm, key_cache_gamma, mse_bits)
attn_scores[t, h, k] = Dot(query_rotated[t, h, :], key_hat[k, kv_h, :])
```

其中 `kv_h = h / (query_heads / key_heads)`。`seq_lens` 之外的位置写为负无穷。

对外仅提供 `aclnnTurboQuantAttentionScore`。内部采用两个 AICore 节点：Prefill 路径先通过
`TqQueryPack` 对 Query 预排布，再由 `TqScoreCompute` 完成解码和打分；Decode 路径只执行
`TqScoreCompute`。两个内部节点不是公开接口。

## 参数说明

| 参数名 | 输入/输出/属性 | Shape | 数据类型 | 格式 |
|---|---|---|---|---|
| query_rotated | 输入 | `[T, Nq, 128]` | BFLOAT16 | ND |
| query_qjl | 输入 | `[T, Nq, 128]` | BFLOAT16 | ND |
| key_cache_idx | 输入 | `[P, 128, Nkv, 16 * mse_bits]` | UINT8 | ND |
| key_cache_qjl | 输入 | `[P, 128, Nkv, 16]` | UINT8 | ND |
| key_cache_norm | 输入 | `[P, 128, Nkv]` | BFLOAT16、FLOAT16 | ND |
| key_cache_gamma | 输入 | `[P, 128, Nkv]`，dtype 与 `key_cache_norm` 相同 | BFLOAT16、FLOAT16 | ND |
| block_table | 输入 | `[B, M]` | INT32 | ND |
| seq_lens | 输入 | `[B]` | INT32 | ND |
| mse_bits | 属性 | 标量，取值 2、3 或 4 | INT64 | - |
| attn_scores | 输出 | `[T, Nq, L]` | FLOAT32 | ND |

`T` 为 Query token 数，`Nq` 为 Query head 数，`Nkv` 为 Key head 数，`P` 为物理 cache block
数，`B` 为 batch，`M` 为每个 batch 的最大逻辑 block 数。

## 约束说明

- 当前仅注册 `ascend910b`，对应 Atlas A2 系列产品。
- 所有 Tensor 必须是连续 ND Tensor。
- head dimension 和 cache block size 均固定为 128。
- `Nq` 必须能被 `Nkv` 整除。
- Decode 路径要求 `T == B`；Prefill 路径要求 `B == 1 && T > 1`。
- `mse_bits` 仅支持 2、3、4。
- 输出最后一维 `L` 必须位于 `[0, M * 128]`。
- `block_table` 中的有效物理页号必须位于 `[0, P)`，`seq_lens[b]` 必须位于 `[0, L]`；运行时遇到小于 0 或大于等于 `P` 的物理页号时，该页输出按无效位置填充为负无穷，且不会访问 cache。
- 输入 cache 必须按本算子规定的连续 bit-packed 布局生成。

## 调用说明

ACLNN 接口、参数含义和两阶段调用方法参见
[`docs/aclnnTurboQuantAttentionScore.md`](docs/aclnnTurboQuantAttentionScore.md)。可运行样例参见
[`examples/test_aclnn_turbo_quant_attention_score.cpp`](examples/test_aclnn_turbo_quant_attention_score.cpp)。

## 编译与测试

```bash
export ASCEND_HOME_PATH=/path/to/cann
source "${ASCEND_HOME_PATH}/set_env.sh"
# 使用已安装构建依赖的 Python 环境，例如：conda activate cann91
bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=turbo_quant_attention_score

bash build.sh --ophost_test --experimental --soc=ascend910b \
    --ops=turbo_quant_attention_score

# 安装上一步生成的 custom 包后执行 ACLNN 示例
bash build.sh --run_example turbo_quant_attention_score eager cust \
    --experimental --soc=ascend910b
```

完整自验覆盖 Decode 4K/64K/128K 和 Prefill 4K/32K。精度判据为有限值最大绝对误差小于
`1e-3`，同时检查屏蔽位置、Score relative L2 和 KL；性能测试使用 20 次预热、100 次计时、
3 轮统计，并在空闲独占设备上执行。

## 目录说明

```text
├── docs/                         # ACLNN 接口文档
├── examples/                     # ACLNN 调用样例
├── op_host/op_api/               # 对外 ACLNN 接口
├── tq_query_pack/                # 内部 Prefill Query 预排布节点
├── tq_score_compute/             # 内部量化 cache 解码与打分节点
└── tests/ut/op_host/             # Host InferShape 和布局契约单元测试
```

## 贡献说明

贡献信息以最终 PR 记录为准。
