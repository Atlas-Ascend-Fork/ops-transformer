# flash_attn测试框架开发说明

修改共享harness、golden、memory、compare、paged路径前，先读本文。

## 0. 读者与范围

本文面向框架维护者和算子适配开发者。

- 用户命令、参数和典型用法见各前端README：`../pytests/README.md`、`../random_fuzz/README.md`、`../perf_benchmark/README.md`；
- 本文按“执行过程 → 分层结构（框架/算子）→ 修改入口 → 约定 → 兼容性”的顺序，说明边界、契约、扩展入口和变更规则；
- 本地最小验证：`python3 -m pytest harness/tests harness/ops/flash_attn/tests perf_benchmark/tests -q`；提交前执行`pre-commit run --files <changed files>`；
- 不覆盖算子数学推导、kernel性能调优和CI基础设施建设。

## 1. 执行过程

### 1.1 入口与阶段

```text
run.py / pytest前端 / random_fuzz / perf_benchmark
└─ harness/runner.py
   ├─ harness/ops/flash_attn/package.py
   ├─ harness/guards.py
   ├─ harness/compare.py
   ├─ harness/perf.py
   └─ harness/results.py
```

阶段固定为：

```text
gen -> golden -> run -> compare
```

perf是`run`的模式，不是独立阶段；该模式只由`perf_benchmark`使用。

### 1.2 阶段职责

| 阶段 | 框架负责 | 算子负责 | 产出 |
|---|---|---|---|
| `gen` | 调度、内存/合法性门禁、输入指纹、可选持久化 | `build_inputs` | `inputs`、digest |
| `golden` | 调度、golden配置/fingerprint、可选持久化 | `golden` | `golden` |
| `run` | 调度、执行模式、可选持久化、perf采集 | `execute` | `npu_out`或perf rows |
| `compare` | 调度、结果记录与写出 | `compare` | metrics、passed |

- `compare`需要`golden`和本次`run`的输出；未计算`golden`时只加载已保存的`golden.pt`，当前不加载`npu_out.pt`。
- 只有`--save-data`时才持久化`inputs.pt`、`golden.pt`、`npu_out.pt`；分阶段执行依赖这些文件。

### 1.3 前端入口与共享参数

- `run.py`把`functional`映射到`pytests/test_flash_attn.py`，`perf`映射到`perf_benchmark/run_perf.py`，`random`映射到`random_fuzz/run_random_fuzz.py`；只解析mode并透传参数。
- 三个工具都通过`harness.Runner`执行。
- `cli_common.py`定义共享参数；`--golden`和`--compare-mode`的默认值/choices分别来自`DEFAULT_GOLDEN_SPEC`和`COMPARE_MODES`。

## 2. 分层结构

### 2.1 框架层

`runner.py`是框架调度核心，负责`gen -> golden -> run -> compare`的阶段顺序、异常分类和结果记录；在`run`阶段按`perf`配置进入性能模式，调用`perf.py`完成性能采集。`runner.py`、`files.py`、`guards.py`是跨阶段文件，阶段表会按使用场景重复列出。

#### 2.1.1 阶段

##### gen

| 文件 | 用途 | 职责 |
|---|---|---|
| `case.py`、`loader.py` | 输入准备 | 解析`case/v1`、构造`Case`、计算稳定`case_id`、按过滤条件筛选 |
| `runner.py` | 阶段调度 | 调用算子`build_inputs`，处理异常和结果记录 |
| `files.py` | 输入持久化 | 计算`inputs_digest`，`--save-data`时保存`inputs.pt`/`inputs_meta.json` |
| `guards.py` | 生成前门禁 | 调用`legalize`，估算host/HBM峰值并判断是否`skip` |

##### golden

| 文件 | 用途 | 职责 |
|---|---|---|
| `runner.py` | 阶段调度 | 调用算子`golden`，处理异常和结果记录 |
| `golden_spec.py` | 参考配置 | 解析device/compute_dtype/p_dtype等字段，生成fingerprint |
| `files.py` | 参考数据存取 | 需要时加载`inputs.pt`，`--save-data`时保存`golden.pt`/`golden.json` |

##### run

| 文件 | 用途 | 职责 |
|---|---|---|
| `runner.py` | 执行调度 | 选择eager/graph/perf模式；graph缺少torchair时直接报错 |
| `files.py` | 输出持久化 | `--save-data`时保存`npu_out.pt` |
| `perf.py` | perf模式采集 | L2 flush、profiler启动/读取、MFU/MBU、roofline理论耗时 |

##### compare

| 文件 | 用途 | 职责 |
|---|---|---|
| `runner.py` | 阶段调度 | 调用精度判定，归类`pass/fail/skip/fatal`并记录指标 |
| `compare.py` | 精度判定 | 两方/三方比较、dtype容差、失败样本统计 |
| `files.py` | 参考数据加载 | 未计算golden时加载已保存的`golden.pt` |
| `results.py` | 结果写出 | 定义列顺序、扁平化算子指标、写唯一`results.csv` |

#### 2.1.2 性能数据采集

| 文件 | 用途 | 职责 |
|---|---|---|
| `perf.py` | 性能采集 | L2 flush、msprof profiler、task time解析、MFU/MBU、roofline理论耗时 |
| `device_spec.py` | 设备峰值 | 按设备型号映射HBM峰值和标准频率；`perf_worker`读取峰值，`backfill`在msprof结束后补标准频率/峰值；不属于主阶段 |
| `perf_benchmark/` | 性能前端 | worker、回填、绘图和性能case生成 |

性能链路：`perf_benchmark/run_perf.py` → `perf_worker.py`（`Runner(perf=True)`）→ msprof → `backfill.py` → `plot_bench.py`；`device_spec.py`为worker和backfill提供峰值。

### 2.2 算子适配层

#### 2.2.1 契约与公共

| 文件 | 用途 | 职责 |
|---|---|---|
| `package.py` | 算子契约 | 暴露`legalize`/`normalize`/`build_inputs`/`golden`/`execute`/`compare`/`perf_context`/`theoretical_workload`等契约，框架只通过它调用算子 |
| `params.py` | 参数归一化 | 补默认值、处理dtype别名、TND/PA派生量、过滤空`seqused`/`cu_seqlens`，保证参数可执行 |
| `memory.py` | 内存估算 | `tensor_footprint`计算张量占用，`golden_footprint`/`estimate`组合阶段峰值 |

#### 2.2.2 gen

| 文件 | 用途 | 职责 |
|---|---|---|
| `inputs.py` | 输入生成 | 按layout/PA生成q/k/v/block_table/sinks；支持范围采样、TND变长和paged page布局 |

#### 2.2.3 golden

| 文件 | 用途 | 职责 |
|---|---|---|
| `golden_truth.py` | CPU/GPU参考 | 行分块参考实现，处理layout转换、PA dense化和paged加载；提供`golden_peak` |
| `golden_npu.py` | NPU参考 | 在设备上执行同一套参考逻辑，供默认`device=npu`使用 |
| `policy.py` | P落盘策略 | 解析`p_block`、`softmax_mode`、`softmax_block_size`，决定P是否分块/落盘 |

#### 2.2.4 run

| 文件 | 用途 | 职责 |
|---|---|---|
| `executor.py` | NPU调用 | 按eager/graph模式调用算子，组装metadata和kernel参数 |
| `layout.py` | layout/PA转换 | BNSD/BSND/TND转换、PA page读取和`PagedKVLoader`，不感知`p_dtype` |

`compare`没有独立算子文件，由`package.compare`分发到通用比较逻辑；`package.py`中的方法按阶段被`runner.py`调用。

### 2.3 边界约束

**三个工具之间**

- `functional`、`random`、`perf`互不import，各自提供独立入口。
- `functional`只由pytest驱动；`random`、`perf`不得进入pytest进程。
- 三者不得各自复制`gen/golden/run/compare`调度。

**三个工具与harness**

- 三者必须通过`harness.Runner`执行，只通过`Runner`、`PACKAGE`、`results`等公共接口交互。
- `perf`的msprof回填和绘图必须在Runner外；`run.py`只转发，不实现业务逻辑。
- `harness`不得import任何前端；`perf_benchmark`不得被pytest前端或`runner.py` import。

**harness与算子**

- 框架不得import算子内部，只通过`PACKAGE`触达算子；算子不得反向import `runner.py`。
- 框架不得出现flash_attn专属字段或分支；算子指标由算子返回，框架只负责扁平化写出。

**其他**

- 修改阶段行为前，先判定属于框架还是算子。
- pytest用例收集、marker和fixture必须留在pytest前端。

## 3. 修改入口

### 3.1 公共

| 要改什么 | 去哪里 |
|---|---|
| 新增case字段 | `ops/flash_attn/params.py` + `package.py`的`legalize` |
| 张量占用 | `ops/flash_attn/memory.py` |
| 内存/合法性门禁 | `harness/guards.py` |
| golden/compare内存行为 | `memory.estimate`、`golden_peak`、guard测试，并同步README/开发文档 |

### 3.2 gen

| 要改什么 | 去哪里 |
|---|---|
| 输入数据/分布 | `ops/flash_attn/inputs.py` |

### 3.3 golden

| 要改什么 | 去哪里 |
|---|---|
| golden数学 | `ops/flash_attn/golden_truth.py`（CPU/参考）、`golden_npu.py`（NPU） |
| golden峰值 | `golden_truth.golden_peak` |
| P精度/块策略 | `ops/flash_attn/policy.py` + `harness/golden_spec.py` |

### 3.4 run

| 要改什么 | 去哪里 |
|---|---|
| NPU调用 | `ops/flash_attn/executor.py` |
| PA/page加载 | `ops/flash_attn/layout.py` |

### 3.5 compare

| 要改什么 | 去哪里 |
|---|---|
| 精度判定 | `harness/compare.py` |
| 结果列/表格 | `harness/results.py`；增删列同步CSV读写和测试 |

### 3.6 性能

| 要改什么 | 去哪里 |
|---|---|
| 性能采集/频率/MFU | `harness/perf.py` |
| 设备峰值/标准频率（仅perf） | `harness/device_spec.py` |
| 性能离线分析 | `perf_benchmark/` |


### 3.7 新增算子适配顺序

1. `package.py`：实现契约方法和`normalize/legalize`；
2. `params.py`：补默认值、派生态和参数校验；
3. `inputs.py`：实现输入生成；
4. `golden_truth.py`/`golden_npu.py`：实现参考；`policy.py`：实现P落盘策略；
5. `executor.py`/`layout.py`：实现执行和layout/PA转换；
6. `memory.py`：实现`tensor_footprint`/`estimate`；
7. 补单测，并同步README/开发文档。

## 4. 约定

### 4.1 内存估算与guard

- 估算在`gen/golden`前完成，必须是纯计算：
  - 不创建NPU tensor；
  - 不执行golden；
  - 不生成大`block_table`。
- `tensor_footprint`只算输入/输出/aux/lse等张量。
- `golden_footprint`只算golden阶段峰值，包含fp32中间量、K/V/k_t、chunk scratch、paged block buffer。
- `persist`必须等于本次`save_data`，不得写死；否则guard会误`skip`或低估内存。

### 4.2 golden精度

`GoldenSpec`字段及取值见`../pytests/README.md`的`--golden`表。

- `p_dtype`、`p_block`、`kv_loader`是相互独立的参数：`p_dtype`控制P落盘精度，`p_block`控制P分块大小，`kv_loader`控制KV加载方式；设置一个不会自动改变另外两个。
- golden精度不能回退；kernel不要求与golden逐位一致，按dtype容差比较。
- 浮点累加保持fp32/fp64，不能为了省内存改成fp16累加。

### 4.3 结果与错误

- `results.csv`只有一份，一次kernel执行一行。
- `status`：`pass` / `fail` / `error` / `skip` / `fatal` / `perf`；`perf`仅出现在未开`--verify`的性能模式。
- 设备/队列致命错误按`fatal`处理，不能当精度失败。
- `random_fuzz`的`failures.json`保留`op`、`case_id`、`tags`、`params`、`status`、`detail`和完整`metrics`，可直接重放。
- `results.csv`每个精度指标只扁平化`_passed`、`_fail_ratio`、`_max_abs`；`fail_cnt`、`samples`、`max_rel`等完整比较指标只在`failures.json`的`metrics`中。
- `results.csv`的核心列和算子扩展列由`harness/results.py`统一定义；新增profiling列只追加，不改变已有列语义。
- 不要为了临时信息随意增加CSV列。

### 4.4 实现约定

- 内存方案变更前先给出峰值增量并确认，不得先实现后补说明。
- 内存估算变更后必须重新跑基线，确认skip集合变化。
- 大输出直接按最终dtype预分配或分段写入，避免“全量fp32→stack/cat→cast”造成峰值叠加。
- fp32中间量（如`out_b`）append后必须释放，避免`stack`时继续占用host/device内存。
- compare循环内不要每块`.item()`，会频繁触发设备同步。
- 非连续NPU tensor大块切片转float可能触发AsStrided问题，必要时先`contiguous`。
- `device_spec.py`的型号键必须使用实际设备名；未知型号不推断峰值。

## 5. 兼容性与版本

- Case JSON使用`case/v1`；`NORM_VERSION`变化会改变稳定`case_id`，必须重新确认基线；
- `GoldenSpec`新字段必须同步更新`fingerprint()`和`describe()`，否则不同配置会被当成同一配置，旧`golden.pt`可能被误用；
- `run.py`的mode名称和参数透传规则属于稳定接口，新增前端不应改变已有模式；
- 修改结果schema、case schema或默认路径时，必须同步用户README、开发文档和测试。
