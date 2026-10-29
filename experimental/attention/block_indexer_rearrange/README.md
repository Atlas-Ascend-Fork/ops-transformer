# BlockIndexerRearrange

## 产品支持情况

| 产品                            | 是否支持 |
| ------------------------------- | :------: |
| <term>Atlas A3系列产品</term>   |    √     |
| <term>Atlas A2系列产品</term>   |    √     |

## 功能说明

- API功能：`BlockIndexerRearrange`算子将稀疏注意力前处理阶段选取的 TopK block 索引（block 级索引）展开为逐 token 的索引。对每个 query token、每个 head，将选中的 block 索引乘以 `block_size` 并加上块内偏移 `[0, 1, ..., block_size-1]`，从而得到连续的 token 级索引；无效位置填充为 `-1`。常用于变长（TND/varlen）场景下稀疏 Attention 的索引重排。

- 计算公式：

    对于第 $t$ 个 query token、第 $n$ 个 head 的第 $j$ 个 block 索引 $\text{idx}[t, n, j]$（$\text{idx} \ge 0$）：

    $$
    \text{topk\_output}[t, n, j \cdot bs + i] = \text{idx}[t, n, j] \cdot bs + i,\quad i \in [0, bs)
    $$

    其中 $bs$ 为 `block_size`。当 $\text{idx}[t, n, j] < 0$（无效 block）或块内 token 超出实际有效 KV 长度时，对应输出填充为 `-1`。

## 参数说明

| 参数名         | 输入/输出/属性 | 描述                                                                 | 数据类型 | 数据格式 |
| -------------- | -------------- | -------------------------------------------------------------------- | -------- | :------: |
| topk_indices   | 输入           | block 级稀疏索引，shape 为 `[T1, N2, blockCount]`，值为 `-1` 表示无效 | INT32    | ND       |
| acc_seq_qlen   | 输入           | query 累积序列长度（前缀和，首元素为 0），shape 为 `[B+1]`           | INT64    | ND       |
| acc_seq_kvlen  | 输入           | KV 累积序列长度（前缀和，首元素为 0），shape 为 `[B+1]`              | INT64    | ND       |
| causal         | 可选属性       | 是否按因果（下三角）方式计算有效 KV 长度，默认值为 `False`           | BOOL     | -        |
| block_size     | 可选属性       | 每个 block 对应的连续 token 数，默认值为 `1`                         | INT64    | -        |
| topk_output    | 输出           | token 级稀疏索引，shape 为 `[T1, N2, blockCount * block_size]`       | INT32    | ND       |

## 约束说明

- 仅支持变长（TND）场景。
- `topk_indices` 与 `topk_output` 均为 INT32。
- `acc_seq_qlen`、`acc_seq_kvlen` 为累积序列长度（前缀和形式，首元素为 0）。
- 有效 KV 长度由 `acc_seq_kvlen` 与 `acc_seq_qlen` 之差、当前 token 在其 batch 内的偏移及 `causal` 共同决定，超出有效范围的位置填充 `-1`。

## 调用说明

本算子 Host 层通过代码仓库标准入口编译生成 `aclnnBlockIndexerRearrange` 单算子 API。编译示例：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=block_indexer_rearrange -j8
```

`--soc` 可选 `ascend910b`（Atlas A2 系列）或 `ascend910_93`（Atlas A3 系列）。

## 目录结构

```text
block_indexer_rearrange/
  CMakeLists.txt                                  # 算子接入仓库构建
  op_host/
    CMakeLists.txt
    block_indexer_rearrange_def.cpp           # OpDef 定义（输入/输出/属性/平台配置）
    block_indexer_rearrange_infershape.cpp    # InferShape / InferDataType
    block_indexer_rearrange_tiling.h          # TilingData 定义
    block_indexer_rearrange_tiling.cpp        # Tiling 与注册
  op_kernel/
    block_indexer_rearrange.cpp               # Ascend C 内核入口与实现
  tests/
    pytest/                                        # pytest 功能验证（CPU golden + NPU 直调）
      test_block_indexer_rearrange_single.py   # 单用例测试主程序
      block_indexer_rearrange_golden.py        # CPU 侧 golden
      block_indexer_rearrange_process.py       # NPU 侧算子调用
      block_indexer_rearrange_paramset.py      # 用例参数配置
      result_compare_method.py                     # 结果精度对比
      test_run.sh                                  # 执行脚本
```

## 测试

参考 [tests/pytest/README.md](./tests/pytest/README.md)。在 `tests/pytest` 目录下激活自定义算子包后执行：

```bash
bash test_run.sh single
```
