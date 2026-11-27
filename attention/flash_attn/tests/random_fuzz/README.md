# random_fuzz：随机flash_attn测试工具

## 1. 概述

`random_fuzz/`是flash_attn的主要随机测试工具。它按批生成随机合法case，执行功能/精度测试，并持续输出结果和失败用例。

主要功能：

- 在算子支持范围内测试随机用例，范围见第6节“随机用例的范围配置”；
- 通过`--max-cases`指定测试数量，或通过`--max-seconds`指定测试时长；
- 精度失败收集与重放；
- 大case和边界长度抽样。

另提供随机case JSON生成，用于检查输入分布或后续重放。

不适用：

- 性能采集：使用`perf_benchmark`；
- 固定case精度回归：使用`functional`；
- 保存inputs/golden/npu_out：random_fuzz不提供。

## 2. 文件结构

```text
random_fuzz/
├── gen_random_cases.py    # 随机case生成器
├── run_random_fuzz.py     # 随机测试执行器
└── README.md
```

## 3. 环境配置

从`attention/flash_attn/tests`目录执行：

```bash
cd attention/flash_attn/tests
source <CANN>/set_env.sh
```

需要可用的TorchNPU环境；graph模式还需要torchair。

## 4. 使用方法

### 4.1 执行随机测试

```bash
cd attention/flash_attn/tests
python3 random_fuzz/run_random_fuzz.py --max-cases=1000 --out-dir=out/random-run1
```

运行条数、时间预算和失败上限通过参数调整，见下表；需要时在命令后追加`--mem-guard-gb=16`或`--graph`。

### 4.2 生成随机用例

```bash
cd attention/flash_attn/tests
python3 random_fuzz/gen_random_cases.py --seed 7 --out random_fuzz/cases/random.json
```

生成条数由`GEN_CASES`控制；该JSON用于检查输入分布或重放，不是执行随机测试的必需步骤。

### 4.3 常用参数

| 参数 | 默认值 | 可用取值/说明 |
|---|---|---|
| `--out-dir` | `tests/out/random-<timestamp>/` | 任意可写目录 |
| `--device-id` | `0` | 整数≥0 |
| `--seed` | `7` | 整数；相同seed和生成配置可复现case形状 |
| `--max-cases` | `0` | 整数≥0；0=不限制，但必须同时设置`--max-seconds>0` |
| `--max-seconds` | `0` | 浮点≥0；0=不限制 |
| `--max-failures` | `0` | 整数≥0；0=不限制 |
| `--mem-guard-gb` | `32` | 浮点≥0；0只关闭host守卫，HBM×0.9守卫仍生效 |
| `--graph` | 关闭 | 布尔开关；需要torchair |
| `--golden` | `device=npu,compute_dtype=fp32,p_dtype=out_dtype` | 逗号分隔的`key=value`；字段和取值见下方 |
| `--compare-mode` | `two_way` | `two_way`或`three_way` |

`--golden`字段：

`--golden`整体默认值为`device=npu,compute_dtype=fp32,p_dtype=out_dtype`；下表“默认值”指显式传入`--golden`但省略该字段时的取值。

| 字段 | 默认值 | 可用取值 | 说明 |
|---|---|---|---|
| `device` | `cpu` | `cpu`、`npu`、`gpu` | 参考计算设备 |
| `compute_dtype` | `fp32` | `fp32`、`fp64` | `fp64`仅CPU，配合`p_dtype=fp32`使用 |
| `p_dtype` | `out_dtype` | `out_dtype`、`fp32` | P落盘精度 |
| `p_block` | `0`（auto） | 整数，0=auto | P落盘/在线归一化的块大小 |
| `softmax_mode` | `auto` | `auto`、`full`、`online` | softmax实现方式 |
| `softmax_block_size` | `0`（auto） | 整数≥0，0=auto | online模式的KV块大小 |
| `kv_loader` | `dense` | `dense`、`paged` | `paged`仅PA使用 |
| `provider` | `default` | `default`、`peer` | 参考实现口径 |

未识别的字段会直接报错。

`--max-cases=0`时必须同时设置`--max-seconds>0`，否则直接报错，避免无限运行。

- `--mem-guard-gb=16`：host峰值超过16 GiB的case跳过，生成期和执行期同时生效；
- `--graph`：图模式执行，首次执行包含建图/编译开销；缺少torchair时直接报错，不回退eager。

## 5. 输出

每次batch结束后写checkpoint：

```text
<out-dir>/
├── results.csv
└── failures.json   # 有失败时
```

常见状态：

- `pass`：精度通过；
- `fail`：精度失败；
- `error`：执行或比对异常；
- `skip`：算子拒绝或守卫跳过；
- `fatal`：设备/队列不可恢复错误。

`failures.json`包含`op`、`tags`、完整`params`、`case_id`、`status`、`detail`和精度指标。

### 5.1 长跑与批大小

执行器按批生成和执行，默认`GEN_CASES_PER_BATCH=1000`条/批。每批结束后把累计结果重写到`results.csv`和`failures.json`。

- 中断发生在批结束后：已完成批次已落盘；
- 中断发生在批中途：当前批已执行但未写盘的case不会保留；`--max-seconds`到点或致命错误退出前会尽量写一次checkpoint；
- 当前不支持断点续跑，重启后从0开始统计并重写`results.csv`；旧`failures.json`不会自动删除，建议换`--out-dir`保留旧结果。

批大小通过`run_random_fuzz.py`顶部常量修改，没有命令行参数：

```python
GEN_CASES_PER_BATCH = 1000
```

调小可减少中断丢失，但会增加CSV重写次数；调大则相反。`--max-cases`只限制总数，不改变批大小。

### 5.2 复现方法

失败用例可直接用functional入口重放：

```bash
python3 -m pytest -q pytests/test_flash_attn.py --cases=<out-dir>/failures.json
```

- 使用相同`--seed`和相同生成配置可复现case形状；
- `failures.json`中的`params.seed`可用于复现单条输入。

## 6. 随机用例的范围配置

随机用例的生成范围、当前取值和对应变量如下：

| 配置项 | 当前取值/范围 | 变量 |
|---|---|---|
| dtype | `fp16`、`bf16` | `DTYPES` |
| Q/OUT layout | `BNSD`、`BSND`、`TND`及混合Q/OUT组合 | `Q_OUT_LAYOUTS` |
| KV layout | 同Q（dense/TND）、`PA_BBND`、`PA_BNBD`、`PA_NZ` | `KV_LAYOUTS`、`PA_LAYOUTS` |
| mask | `0`、`3`、`4`；`4`支持window | `MASK_MODES` |
| D/DV | `(64,64)`、`(72,72)`、`(128,128)`、`(192,128)`、`(256,256)` | `SUPPORTED_DV` |
| seqlen场景 | `S1_varlen`、`S2_trunc`、`S3_varlen_trunc`、`S4_exact`、`S5_empty`、`S6_pa_decode`、`S7_tnd_pa_varlen` | `SEQ_SCENES` |
| 长度/batch/头数 | Q/KV`0..1M`、B`1..512`、N2`1..16`、G`1..32`、decode S1`1..16` | `Q_SEQ_RANGE`、`KV_SEQ_RANGE`、`BATCH_RANGE`、`N2_RANGE`、`G_RANGE`、`DECODE_S1_RANGE` |
| seqlen采样 | 20%边界+80%log-uniform；pad≤256；总token≤1M | `SEQ_BOUNDARY_RATE`、`NORMAL_LOG_MIN`、`PAD_MAX`、`MAX_TOTAL_TOKENS` |
| scale | default、inv_sqrt_d、`(0,1]`、`(1,FP32_MAX)` | `SCALE_STRATEGIES` |
| window | both_inf、left_inf、right_inf、boundary、uniform_range、overflow_left/right/both | `WINDOW_STRATEGIES`、`WINDOW_CANDIDATES` |
| PA block大小 | `16..1024`，步长16 | `BLOCK_SIZE_RANGE` |
| 可选输入 | `return_softmax_lse`、`sinks`、显式`max_seqlen_q/kv`、`head_dim_v` | `RETURN_SOFTMAX_LSE`、`SINKS`、`EMIT_MAX_SEQLEN_Q`、`EMIT_MAX_SEQLEN_KV`、`EMIT_HEAD_DIM_V` |
| 大case | `>=128K`，1% | `LARGE_SEQ_THRESHOLD`、`LARGE_CASE_RATE` |
| 重试/内存守卫 | 每case重试上限100、安全系数1.2 | `DEFAULT_ATTEMPTS_PER_CASE`、`GEN_SAFETY_FACTOR` |
| 生成条数 | 默认1000 | `GEN_CASES` |
| 执行批大小 | 默认1000 | `GEN_CASES_PER_BATCH` |

除`GEN_CASES_PER_BATCH`位于`run_random_fuzz.py`顶部外，其余变量均位于`gen_random_cases.py`顶部。执行模式默认为eager，`--graph`切换为graph。

## 7. 已知说明与约束

- random_fuzz只做功能/精度，不采集性能；
- 不保存`inputs/golden/npu_out`，没有`--save-data`；
- 不提供`--case-id`/`--testset`/`--skip-cases`，因为case是随机生成的；需要过滤时先落盘JSON，再用functional入口；
- 设备致命错误按`fatal`记录，连续20条fatal才中止整轮；
- 生成流程为：选择场景→采样参数→合法性/内存检查→不合法则重试→执行并写结果；达到尝试上限时返回已有case并报告缺口；
- 生成可能少于请求条数，不会因单个候选失败中断整轮。
