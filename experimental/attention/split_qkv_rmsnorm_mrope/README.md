# SplitQkvRmsnormMrope

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

`SplitQkvRmsnormMrope` 面向 Qwen3-VL 注意力输入处理，在一个 AICore Kernel 中完成融合
QKV/Gate 拆分、Q/K RMSNorm、仿射和三轴 MRoPE，V/Gate 保持原始 BF16 位表示直通。

对每个 Q/K head 向量 `x`：

```text
u = x / sqrt(sum(x * x) / 256 + eps)
a = u * weight + bias
out[0:32]  = a[0:32] * cos - a[32:64] * sin
out[32:64] = a[32:64] * cos + a[0:32] * sin
out[64:256] = a[64:256]
```

无 bias 时省略加法。MRoPE 仅作用于前 64 维，将 `[0,32)` 与 `[32,64)` 组成 32 个旋转维度对。
对第 `j` 个维度对（`0 <= j < 32`），cos/sin 始终使用对应轴输入中的全局下标 `j`：

- `interleaved=false`：连续 section 映射，`j∈[0,11)` 使用轴 0，`j∈[11,22)` 使用轴 1，
  `j∈[22,32)` 使用轴 2，即 section 为 `[11,11,10]`；
- `interleaved=true`：交错映射，使用轴 `j % 3`，即三个轴按维度对轮询。单 token、无 bias、
  `has_gate=true` 的优化路径仅用于该交错模式，并与通用路径采用相同映射。

## 参数说明

| 参数名 | 输入/输出/属性 | Shape | 数据类型 | 格式 |
|---|---|---|---|---|
| qkv | 输入 | `[T, (has_gate ? 2 : 1) * Nq * 256 + 2 * Nkv * 256]` | BFLOAT16 | ND |
| q_weight | 输入 | `[256]` | BFLOAT16 | ND |
| k_weight | 输入 | `[256]` | BFLOAT16 | ND |
| cos_sin | 输入 | `[3, T, 64]` | BFLOAT16 | ND |
| q_bias | 可选输入 | `[256]` | BFLOAT16 | ND |
| k_bias | 可选输入 | `[256]` | BFLOAT16 | ND |
| num_q_heads | 属性 | 当前仅支持 16 | INT64 | - |
| num_kv_heads | 属性 | 当前支持 2、4 | INT64 | - |
| eps | 属性 | 有限正数，默认 `1e-6` | FLOAT | - |
| interleaved | 属性 | `true` 为按 `j % 3` 轮询三轴，`false` 为连续 section `[11,11,10]` | BOOL | - |
| has_gate | 属性 | 输入是否包含 Gate | BOOL | - |
| q_output | 输出 | `[T, Nq * 256]` | BFLOAT16 | ND |
| k_output | 输出 | `[T, Nkv * 256]` | BFLOAT16 | ND |
| v_output | 输出 | `[T, Nkv * 256]` | BFLOAT16 | ND |
| gate_output | 输出 | `[T, has_gate ? Nq * 256 : 0]` | BFLOAT16 | ND |

## 约束说明

- 当前仅注册 `ascend910b`，对应 Atlas A2 系列产品。
- 输入和输出必须为连续 ND BF16 Tensor。
- token 数必须为正数，head dimension 固定为 256，MRoPE dimension 固定为 64。
- `num_q_heads=16`，`num_kv_heads` 取 2 或 4。
- `q_weight`、`k_weight` 以及存在时的 bias 必须为 `[256]`。
- `cos_sin` 必须为 `[3,T,64]`。
- `qkv` 宽度必须与 head 数及 `has_gate` 一致。
- `eps` 必须为有限正数。
- `has_gate=true` 时，Q/Gate 区域按 head 交错排列：`[q0, gate0, q1, gate1, ...]`，随后为连续的 K、V 区域。

## 调用说明

ACLNN 接口参见 [`docs/aclnnSplitQkvRmsnormMrope.md`](docs/aclnnSplitQkvRmsnormMrope.md)，完整示例参见
[`examples/test_aclnn_split_qkv_rmsnorm_mrope.cpp`](examples/test_aclnn_split_qkv_rmsnorm_mrope.cpp)。

## 编译与测试

```bash
export ASCEND_HOME_PATH=/path/to/cann
source "${ASCEND_HOME_PATH}/set_env.sh"

bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=split_qkv_rmsnorm_mrope

bash build.sh --ophost_test --experimental --soc=ascend910b \
    --ops=split_qkv_rmsnorm_mrope
```

验收矩阵包含 Decode、Small、Medium 和 Prefill 共 7 个固定 case。Q/K 最大绝对误差要求小于
`1e-3`，V/Gate 要求 BF16 原始字节一致。

## 贡献说明

贡献信息以最终 PR 记录为准。
