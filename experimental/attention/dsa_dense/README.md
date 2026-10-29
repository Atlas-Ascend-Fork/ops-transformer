# DsaDense（Experimental）

## 产品支持情况

| 产品                          | 是否支持 |
| ----------------------------- | :------: |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |

## 功能说明

`DsaDense` 是 DSA（DeepSeek Sparse Attention）类结构的前向 dense 注意力前处理反向算子。
基于主注意力（main attention）与 indexer 两条通路的 softmax `max`/`sum` 统计量，计算：

- `loss`：main 分布 `P` 与 indexer 分布 `Y` 的 KL 散度（每 batch 求和后累加）；
- `dq` / `dk`：indexer query/key 的梯度；
- `dw`：indexer 逐 head 权重的梯度。

输入采用变长 TND 布局。

## 参数说明

| 参数名     | 输入/输出/属性 | 描述                                              | 数据类型 | 格式 |
| ---------- | -------------- | ------------------------------------------------- | -------- | :--: |
| query      | 输入           | `[T1, N, D]`                                      | BF16     | ND   |
| key        | 输入           | `[T2, N, D]`                                      | BF16     | ND   |
| max        | 输入           | main 注意力 softmax 行最大值，`[N, T1]`           | FLOAT    | ND   |
| sum        | 输入           | main 注意力 softmax 行求和，`[N, T1]`             | FLOAT    | ND   |
| qLen       | 输入           | query 累积序列长度（前缀和），`[B+1]`             | INT64    | ND   |
| kvLen      | 输入           | KV 累积序列长度（前缀和），`[B+1]`                | INT64    | ND   |
| mask       | 输入           | 因果掩码，`[128, 128]`                            | BOOL     | ND   |
| queryIndex | 输入           | indexer query，`[T1, NI, DI]`                     | BF16     | ND   |
| keyIndex   | 输入           | indexer key，`[T2, DI]`                           | BF16     | ND   |
| weights    | 输入           | indexer 逐 head 权重，`[T1, NI]`                  | FLOAT    | ND   |
| maxIndex   | 输入           | indexer softmax 行最大值，`[T1]`                  | FLOAT    | ND   |
| sumIndex   | 输入           | indexer softmax 行求和，`[T1]`                    | FLOAT    | ND   |
| sparse_mode| 属性           | 稀疏模式，仅支持 `1`（causal）                    | INT      | -    |
| scale      | 属性           | 缩放因子，`1/sqrt(D)`                             | FLOAT    | -    |
| loss       | 输出           | KL loss，`[1]`                                    | FLOAT    | ND   |
| dq         | 输出           | indexer query 梯度，`[T1, NI, DI]`                | BF16     | ND   |
| dk         | 输出           | indexer key 梯度，`[T2, DI]`                      | BF16     | ND   |
| dw         | 输出           | indexer weights 梯度，`[NI, T1]`                  | FLOAT    | ND   |

## 约束说明

- 仅支持变长（TND）布局与因果（causal）模式（`sparse_mode=1`）。
- `mask` 形状固定为 `[128, 128]`。
- `NI <= 128`。

## 构建

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=dsa_dense -j8
```

`--soc` 可选 `ascend910b`（Atlas A2 系列）或 `ascend910_93`（Atlas A3 系列）。

## 目录结构

```text
dsa_dense/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    dsa_dense_def.cpp          # OpDef
    dsa_dense_infershape.cpp   # InferShape / InferDataType
    dsa_dense_tiling.h         # TilingData 定义
    dsa_dense_tiling.cpp       # Tiling 与注册（含 matmul lib tiling）
  op_kernel/
    dsa_dense.cpp              # Ascend C 内核入口
    dsa_dense.h                # 内核实现
  tests/
    pytest/                        # pytest 功能验证（CPU golden + NPU 直调）
      test_dsa_dense_single.py
      dsa_dense_golden.py
      dsa_dense_process.py
      dsa_dense_paramset.py
      result_compare_method.py
      test_run.sh
```

## 测试

参考 [tests/pytest/README.md](./tests/pytest/README.md)。在 `tests/pytest` 目录下激活自定义算子包后执行：

```bash
bash test_run.sh single
```
