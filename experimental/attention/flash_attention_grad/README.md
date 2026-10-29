# FlashAttentionGrad（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |

## 功能说明

`FlashAttentionGrad` 是变长（TND）FlashAttention 反向算子，给定输出梯度 `dy` 与前向中间量
`attention_out`/`softmax_lse`，计算 query/key/value 的梯度 `dq`/`dk`/`dv`。支持因果（causal）与非因果模式。

## 参数说明

| 参数名           | 输入/输出/属性 | 描述                                              | 数据类型      | 格式 |
| ---------------- | -------------- | ------------------------------------------------- | ------------- | :--: |
| dy               | 输入           | 输出梯度，`[T1, N1, D]`                           | FLOAT16 / BF16| ND   |
| query            | 输入           | `[T1, N1, D]`                                     | FLOAT16 / BF16| ND   |
| key              | 输入           | `[T2, N2, D]`                                     | FLOAT16 / BF16| ND   |
| value            | 输入           | `[T2, N2, VD]`                                    | FLOAT16 / BF16| ND   |
| attention_out    | 输入           | 前向输出，`[T1, N1, VD]`                          | FLOAT16 / BF16| ND   |
| softmax_lse      | 输入           | 前向 lse，`[N1, T1]`（NT 布局）                   | FLOAT         | ND   |
| actual_seq_qlen  | 可选输入       | query 累积序列长度，`[B+1]`                       | INT64         | ND   |
| actual_seq_kvlen | 可选输入       | KV 累积序列长度，`[B+1]`                          | INT64         | ND   |
| prm_mask         | 可选输入       | 前缀掩码                                          | BOOL          | ND   |
| sink             | 可选输入       | sink 项                                           | FLOAT16 / BF16| ND   |
| scale_value      | 可选属性       | 缩放因子，默认 `1.0`                              | FLOAT         | -    |
| pre_tockens/next_tockens | 可选属性 | 前/后向窗口，默认 `INT32_MAX`                    | INT           | -    |
| head_num         | 必选属性       | head 数                                           | INT           | -    |
| input_layout     | 必选属性       | 布局，仅支持 `TND`（默认）                        | STRING        | -    |
| sparse_mode      | 可选属性       | 稀疏模式，默认 `0`                                | INT           | -    |
| dq               | 输出           | query 梯度，`[T1, N1, D]`                         | FLOAT16 / BF16| ND   |
| dk               | 输出           | key 梯度，`[T2, N2, D]`                           | FLOAT16 / BF16| ND   |
| dv               | 输出           | value 梯度，`[T2, N2, VD]`                        | FLOAT16 / BF16| ND   |

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=flash_attention_grad -j8
```

## 目录结构

```text
flash_attention_grad/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    flash_attention_grad_def.cpp
    flash_attention_grad_tiling.cpp                # Tiling 入口与注册
    flash_attention_grad_tiling.h / _common.h          # TilingData / 公共定义
    flash_attention_score_grad_tiling_s1s2_bn2gs1s2.*  # tiling 模板
    flash_attention_grad_tiling_s_bn2gs1s2.*       # tiling 模板
    flash_attention_grad_tiling_registry.cpp       # TilingRegistry 实现
    white_list.h
    base/                                               # tiling 共享基础
  op_kernel/
    flash_attention_grad.cpp / *.h                  # Ascend C 内核
  tests/
    pytest/                                             # pytest 功能验证
```

## 测试

参考 [tests/pytest/README.md](./tests/pytest/README.md)。
