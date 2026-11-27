# perf_benchmark：flash_attn性能测试

## 1. 概述

`perf_benchmark/`是flash_attn的性能测试工具，用msprof采集kernel执行时间，输出`results.csv`并生成性能曲线。

主要功能：

- 指定case集的性能测试和性能数据收集；
- 测试固定参数下的batch线性度、S2负载均衡性，以及不同S1长度下MFU/MBU/duration随S2变化的趋势图。

不适用：

- 固定case精度回归：使用`pytests`；
- 随机功能测试：使用`random_fuzz`。

## 2. 文件结构

```text
perf_benchmark/
├── run_perf.py            # 执行入口
├── perf_worker.py         # 单case性能采集
├── gen_bench_cases.py     # YAML配置生成case
├── backfill.py            # msprof结果回填
├── plot_bench.py          # 绘图
├── bench_config.yaml      # 默认测试配置
└── README.md
```

## 3. 环境配置

从`attention/flash_attn/tests`目录执行：

```bash
cd attention/flash_attn/tests
source <CANN>/set_env.sh
```

需要可用的TorchNPU环境、msprof和matplotlib；graph模式还需要torchair。

## 4. 使用方法

### 4.1 常见用法与指令

#### 4.1.1 配置驱动

```bash
python3 perf_benchmark/run_perf.py --config=perf_benchmark/bench_config.yaml --perf-runs=5 --out-dir=out/perf-run1
```

配置驱动会自动生成case、执行msprof、回填结果并绘图。生成结果写入`<out-dir>/cases/`，图写入`<out-dir>/figs/`。

只检查配置和case数量：

```bash
python3 perf_benchmark/gen_bench_cases.py --dry-run
```

#### 4.1.2 指定case

```bash
python3 perf_benchmark/run_perf.py --cases=path/to/cases.json --perf-runs=5 --out-dir=out/perf-run1
```

`--cases`可重复传入多个JSON文件或目录；该入口不生成case，也不绘图。过滤后没有case会直接报错。

### 4.2 常用参数

| 参数 | 默认值 | 可用取值/说明 |
|---|---|---|
| `--config` | 无 | YAML路径；与`--cases`互斥且必填其一 |
| `--cases` | 无 | JSON文件或目录，可重复；与`--config`互斥且必填其一 |
| `--perf-runs` | `5` | 整数≥1；正式测量轮数 |
| `--out-dir` | `tests/out/perf-<timestamp>/` | 任意可写目录 |
| `--verify` | 关闭 | 布尔开关；性能测量后追加精度校验 |
| `--golden` | `device=npu,compute_dtype=fp32,p_dtype=out_dtype` | 仅`--verify`生效；字段和默认值见4.3 |
| `--compare-mode` | `two_way` | `two_way`或`three_way`；仅`--verify`生效 |
| `--case-id` | 不过滤 | 逗号分隔的模式；支持子串和通配符 |
| `--testset` | 不过滤 | 逗号分隔的tag |
| `--skip-cases` | 不跳过 | `.txt`文件或逗号分隔的模式 |
| `--mem-guard-gb` | `32` | 浮点≥0；0只关闭host守卫，HBM×0.9守卫仍生效 |
| `--graph` | 关闭 | 布尔开关；需要torchair |
| `--device-id` | `0` | 整数≥0 |

`--verify`关闭时只采集性能，`status=perf`；开启后状态为`pass/fail/skip/error`，并写入精度字段。

资源与执行控制：

- `--mem-guard-gb=16`：host峰值超过16 GiB的case跳过；
- `--graph`：图模式执行；缺少torchair时直接报错，不回退eager。

### 4.3 `--golden`配置

`--golden`定义`--verify`时的参考计算方式，不影响msprof性能采集；未开启`--verify`时忽略。默认在NPU上计算参考，fp32累加，P落盘到V的dtype；需要高精度真值时使用`device=cpu,compute_dtype=fp64,p_dtype=fp32`。

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

## 5. 输出

配置驱动输出：

```text
<out-dir>/
├── cases/
│   ├── bench_cases.json
│   └── bench_manifest.json
├── profiling/
│   └── PROF_*/
├── results.csv
├── perf_manifest.json
└── figs/
```

指定case入口不写`cases/`和`figs/`，仍写`results.csv`、`perf_manifest.json`和`profiling/`。

`figs/`中的图：

| 文件 | 说明 |
|---|---|
| `s2_curve_duration_log2.png`、`s2_curve_duration_linear.png` | duration随S2变化；按S1子图、按layout画线，横轴分别为log2/linear |
| `s2_curve_mfu_log2.png`、`s2_curve_mfu_linear.png` | MFU随S2变化；实线为实际频率，虚线为标准频率 |
| `s2_curve_mbu_log2.png`、`s2_curve_mbu_linear.png` | MBU随S2变化 |
| `s2_curve_combined_log2.png`、`s2_curve_combined_linear.png` | 实际/标准MFU与MBU画在同一0–100%纵轴 |
| `batch_linearity.png` | duration随B变化；按phase子图、按layout画线 |
| `s2_balance.png` | 不同distribution的duration柱状图，按layout分组 |

`results.csv`主要字段：

| 字段 | 含义 |
|---|---|
| `duration_us` | 本次kernel执行时间；不含host编译和调度 |
| `flops` / `bytes` | 理论浮点运算量/访存量 |
| `theory_us` / `theory_bound` | 实际频率roofline耗时和瓶颈（`cube`/`bandwidth`） |
| `theory_us_std` / `theory_bound_std` | 标准频率roofline耗时和瓶颈 |
| `mfu_pct` / `mfu_std_pct` | 实际频率/标准频率算力利用率 |
| `mbu_pct` | 带宽利用率 |
| `aic_freq_mhz` / `aiv_freq_mhz` | 本次run推导的AIC/AIV频率 |
| `aic_core_num` / `aiv_core_num` | AIC/AIV核数 |
| `device_model` / `standard_freq_mhz` / `standard_peak_gbs` | 设备型号和标准峰值 |
| `tiling_key` | kernel tiling标识 |

`tiling_key`之后继续追加msprof的block数、shape/dtype/format、AIC/AIV分项耗时和利用率等列；未采集到的字段留空。

重绘性能图：

```bash
python3 perf_benchmark/plot_bench.py --results=out/perf-run1/results.csv --manifest=out/perf-run1/cases/bench_manifest.json --out-dir=out/perf-run1/figs
```

`perf_manifest.json`记录case/run顺序和设备峰值；`cases/bench_manifest.json`记录实验配置和case元数据。`profiling/`保留原始msprof数据，便于重新回填或分析。

## 6. 性能用例的范围配置

配置位于`perf_benchmark/bench_config.yaml`。顶层`version`必须为1；`common`除`phase_threshold`外均必填，`phase_threshold`缺省16。

### 6.1 公共配置

| 配置项 | 当前取值/约束 | 说明 |
|---|---|---|
| `common.dtype` | `fp16` | 数据类型 |
| `common.layouts` | `BSND`、`BNSD` | Q/KV/OUT layout；不能为空 |
| `common.D` / `common.DV` | `128` / `128` | 头维 |
| `common.N1` / `common.N2` | `32` / `4` | Q/KV头数；`N1`必须是`N2`的整数倍 |
| `common.mask_mode` | `3` | mask类型 |
| `common.win_left` / `common.win_right` | `2147483647` | window边界 |
| `common.scale` | `inv_sqrt_d` | `inv_sqrt_d`或正浮点数 |
| `common.return_softmax_lse` | `false` | 是否返回lse |
| `common.phase_threshold` | `16` | decode/prefill分界；可省略 |
| KV | dense | 不带`cu_seqlens`，非PA |

### 6.2 s2_curve

| 配置项 | 当前取值 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否启用 |
| `batch` | `8` | batch大小 |
| `s1` | `[1, 3, 8, 16, 128, 256, 512, 2048, 8192]` | Q长度列表 |
| `s2_aligned` | `[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]` | S2对齐点 |
| `s2_tail_sizes` | `[1, 16, 32, 64, 127]` | 对齐点后追加的尾块 |

### 6.3 batch_linearity

| 配置项 | 当前取值 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否启用 |
| `batch_values` | `[1, 2, 4, 8, 16, 32, 64]` | B扫描值 |
| `shapes` | prefill`S1=4096,S2=6144`；decode`S1=1,S2=24576` | 按phase设置形状 |

### 6.4 s2_balance

| 配置项 | 当前取值 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否启用 |
| `batch` | `16` | batch大小 |
| `S2` | `8192` | KV总长度 |
| `fill` | `0.5` | 有效长度填充比例 |
| `distributions` | `[uniform, ramp, skew, outlier, alternating]` | batch内长度分布 |
| `shapes` | prefill`S1=4096`；decode`S1=1` | 按phase设置形状 |

`experiments`缺省段视为关闭，存在时`enabled`缺省`true`；未知字段直接报错。

设备峰值自动按型号映射：已知型号提供HBM峰值和标准频率；未知型号不猜测峰值，对应指标留空。

## 7. 已知说明与约束

- 结果表头与`pytests`、`random_fuzz`一致，未采集的精度或profiling列留空；
