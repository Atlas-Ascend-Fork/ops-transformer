# FlashAttention（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |

## 功能说明

`FlashAttention` 是变长（TND）FlashAttention 前向算子，支持因果（causal）与非因果模式，
输出注意力结果 `attention_out`、`lse_out`（可按 `NT` 布局），以及 `softmax_max`/`softmax_sum`/`max_logits`。

## 参数说明

| 参数名            | 输入/输出/属性 | 描述                                                   | 数据类型      | 格式 |
| ----------------- | -------------- | ------------------------------------------------------ | ------------- | :--: |
| query             | 输入           | `[T1, N1, D]`                                          | FLOAT16 / BF16| ND   |
| key               | 输入           | `[T2, N2, D]`                                          | FLOAT16 / BF16| ND   |
| value             | 输入           | `[T2, N2, VD]`                                         | FLOAT16 / BF16| ND   |
| actual_seq_qlen   | 可选输入       | query 累积序列长度，`[B+1]`                            | INT64         | ND   |
| actual_seq_kvlen  | 可选输入       | KV 累积序列长度，`[B+1]`                               | INT64         | ND   |
| atten_mask        | 可选输入       | 注意力掩码                                             | BOOL          | ND   |
| prm_mask          | 可选输入       | 前缀掩码                                               | BOOL          | ND   |
| sink              | 可选输入       | sink 项                                                | FLOAT16 / BF16| ND   |
| scale_value       | 可选属性       | 缩放因子，默认 `1.0`                                   | FLOAT         | -    |
| pre_tockens       | 可选属性       | 前向窗口，默认 `INT32_MAX`                             | INT           | -    |
| next_tockens      | 可选属性       | 后向窗口，默认 `INT32_MAX`                             | INT           | -    |
| head_num          | 必选属性       | head 数                                                | INT           | -    |
| input_layout      | 必选属性       | 布局，仅支持 `TND`（默认）                             | STRING        | -    |
| sparse_mode       | 可选属性       | 稀疏模式，默认 `0`                                     | INT           | -    |
| softmax_max       | 输出           | softmax 行最大值                                       | FLOAT         | ND   |
| softmax_sum       | 输出           | softmax 行求和                                         | FLOAT         | ND   |
| lse_out           | 输出           | log-sum-exp                                            | FLOAT         | ND   |
| attention_out     | 输出           | 注意力结果                                             | FLOAT16 / BF16| ND   |
| max_logits        | 输出           | 最大 logits                                            | FLOAT         | ND   |

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=flash_attention -j8
```

## 目录结构

```text
flash_attention/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    flash_attention_def.cpp                 # OpDef
    flash_attention_infershape.cpp          # InferShape / InferDataType
    flash_attention_tiling.h                # TilingData 定义
    flash_attention_tiling.cpp              # Tiling 入口与注册
    flash_attention_tiling_general.cpp      # Tiling 实现
    flash_attention_tiling_registry.cpp     # template registry（TilingRegistry 实现）
    base/                                        # tiling 共享基础（tiling_base / registry / copy-transpose）
  op_kernel/
    flash_attention.cpp / .h / _common.h     # Ascend C 内核
  tests/
    pytest/                                      # pytest 功能验证
```

## 测试

参考 [tests/pytest/README.md](./tests/pytest/README.md)。
