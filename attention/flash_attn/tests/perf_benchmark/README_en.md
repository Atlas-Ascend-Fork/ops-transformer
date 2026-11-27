# perf_benchmark: flash_attn performance tests

## 1. Overview

`perf_benchmark/` is the performance test tool for flash_attn. It uses msprof to collect kernel execution time, writes `results.csv`, and generates performance curves.

Main features:

- performance testing and performance data collection for specified case sets;
- batch linearity and S2 balance under fixed parameters, plus trend plots of MFU/MBU/duration versus S2 at different S1 lengths.

Not for:

- fixed-case precision regression: use `pytests`;
- random functional tests: use `random_fuzz`.

## 2. Files

```text
perf_benchmark/
├── run_perf.py            # entry point
├── perf_worker.py         # per-case performance collection
├── gen_bench_cases.py     # generate cases from YAML
├── backfill.py            # backfill msprof results
├── plot_bench.py          # plotting
├── bench_config.yaml      # default test config
└── README.md
```

## 3. Environment

Run from `attention/flash_attn/tests`:

```bash
cd attention/flash_attn/tests
source <CANN>/set_env.sh
```

A working TorchNPU environment, msprof, and matplotlib are required. Graph mode also requires torchair.

## 4. Usage

### 4.1 Common usage and commands

#### 4.1.1 Config-driven

```bash
python3 perf_benchmark/run_perf.py --config=perf_benchmark/bench_config.yaml --perf-runs=5 --out-dir=out/perf-run1
```

The config-driven entry generates cases, runs msprof, backfills results, and plots. Generated cases go to `<out-dir>/cases/`; figures go to `<out-dir>/figs/`.

Check only the config and case counts:

```bash
python3 perf_benchmark/gen_bench_cases.py --dry-run
```

#### 4.1.2 Specified cases

```bash
python3 perf_benchmark/run_perf.py --cases=path/to/cases.json --perf-runs=5 --out-dir=out/perf-run1
```

`--cases` may be repeated with multiple JSON files or directories. This entry does not generate cases and does not plot. An empty selection is an error.

### 4.2 Common options

| Option | Default | Allowed values / notes |
|---|---|---|
| `--config` | none | YAML path; mutually exclusive with `--cases` and required |
| `--cases` | none | JSON file or directory; repeatable; mutually exclusive with `--config` and required |
| `--perf-runs` | `5` | integer >= 1; number of measured runs |
| `--out-dir` | `tests/out/perf-<timestamp>/` | any writable directory |
| `--verify` | off | boolean; run precision verification after performance measurement |
| `--golden` | `device=npu,compute_dtype=fp32,p_dtype=out_dtype` | only with `--verify`; fields and defaults are listed in 4.3 |
| `--compare-mode` | `two_way` | `two_way` or `three_way`; only with `--verify` |
| `--case-id` | no filter | comma-separated patterns; substring and wildcard matching |
| `--testset` | no filter | comma-separated tags |
| `--skip-cases` | no skip | `.txt` file or comma-separated patterns |
| `--mem-guard-gb` | `32` | float >= 0; 0 disables only the host guard, the HBM x 0.9 guard remains active |
| `--graph` | off | boolean; requires torchair |
| `--device-id` | `0` | integer >= 0 |

Without `--verify`, only performance is collected and `status=perf`. With `--verify`, the status is `pass/fail/skip/error` and precision fields are filled.

Resource and execution control:

- `--mem-guard-gb=16`: cases whose estimated host peak exceeds 16 GiB are skipped;
- `--graph`: graph execution; missing torchair raises an error instead of falling back to eager.

### 4.3 `--golden` configuration

`--golden` selects the reference computation used by `--verify`; it does not affect msprof performance collection and is ignored without `--verify`. By default the reference runs on NPU with fp32 accumulation and casts P to the V dtype. For a high-precision reference, use `device=cpu,compute_dtype=fp64,p_dtype=fp32`.

`--golden` as a whole defaults to `device=npu,compute_dtype=fp32,p_dtype=out_dtype`; in the table, Default is the value used when an explicit `--golden` omits that field.

| Field | Default | Allowed values | Notes |
|---|---|---|---|
| `device` | `cpu` | `cpu`, `npu`, `gpu` | reference device |
| `compute_dtype` | `fp32` | `fp32`, `fp64` | `fp64` is CPU-only and is used with `p_dtype=fp32` |
| `p_dtype` | `out_dtype` | `out_dtype`, `fp32` | P storage precision |
| `p_block` | `0` (auto) | integer, 0=auto | P storage / online normalization block size |
| `softmax_mode` | `auto` | `auto`, `full`, `online` | softmax implementation |
| `softmax_block_size` | `0` (auto) | integer >= 0, 0=auto | KV block size for online mode |
| `kv_loader` | `dense` | `dense`, `paged` | `paged` is only used for PA |
| `provider` | `default` | `default`, `peer` | reference implementation semantics |

Unknown fields raise an error.

## 5. Output

Config-driven output:

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

The specified-case entry does not write `cases/` or `figs/`; it still writes `results.csv`, `perf_manifest.json`, and `profiling/`.

Figures under `figs/`:

| File | Description |
|---|---|
| `s2_curve_duration_log2.png`, `s2_curve_duration_linear.png` | duration versus S2; subplots by S1 and lines by layout, with log2/linear x-axis |
| `s2_curve_mfu_log2.png`, `s2_curve_mfu_linear.png` | MFU versus S2; solid lines use actual frequency, dashed lines use standard frequency |
| `s2_curve_mbu_log2.png`, `s2_curve_mbu_linear.png` | MBU versus S2 |
| `s2_curve_combined_log2.png`, `s2_curve_combined_linear.png` | actual/standard MFU and MBU on one 0–100% axis |
| `batch_linearity.png` | duration versus B; subplots by phase and lines by layout |
| `s2_balance.png` | duration bars for each distribution, grouped by layout |

Main `results.csv` fields:

| Field | Meaning |
|---|---|
| `duration_us` | kernel execution time for this run; excludes host compile and scheduling |
| `flops` / `bytes` | theoretical floating-point work / memory traffic |
| `theory_us` / `theory_bound` | roofline time and bottleneck (`cube`/`bandwidth`) at actual frequency |
| `theory_us_std` / `theory_bound_std` | roofline time and bottleneck at standard frequency |
| `mfu_pct` / `mfu_std_pct` | compute utilization at actual/standard frequency |
| `mbu_pct` | bandwidth utilization |
| `aic_freq_mhz` / `aiv_freq_mhz` | AIC/AIV frequency derived from this run |
| `aic_core_num` / `aiv_core_num` | AIC/AIV core counts |
| `device_model` / `standard_freq_mhz` / `standard_peak_gbs` | device model and standard peaks |
| `tiling_key` | kernel tiling identifier |

After `tiling_key`, the table appends msprof columns for block counts, shape/dtype/format, AIC/AIV per-pipe time, and utilization. Missing values are left empty.

Replot performance figures:

```bash
python3 perf_benchmark/plot_bench.py --results=out/perf-run1/results.csv --manifest=out/perf-run1/cases/bench_manifest.json --out-dir=out/perf-run1/figs
```

`perf_manifest.json` records case/run order and device peaks; `cases/bench_manifest.json` records experiment configuration and case metadata. `profiling/` keeps the raw msprof data for re-backfill or analysis.

## 6. Performance case scope and configuration

Configuration is in `perf_benchmark/bench_config.yaml`. The top-level `version` must be 1. All `common` fields except `phase_threshold` are required; `phase_threshold` defaults to 16.

### 6.1 Common configuration

| Item | Current values / constraints | Notes |
|---|---|---|
| `common.dtype` | `fp16` | data type |
| `common.layouts` | `BSND`, `BNSD` | Q/KV/OUT layouts; must not be empty |
| `common.D` / `common.DV` | `128` / `128` | head dimensions |
| `common.N1` / `common.N2` | `32` / `4` | Q/KV head counts; `N1` must be a multiple of `N2` |
| `common.mask_mode` | `3` | mask type |
| `common.win_left` / `common.win_right` | `2147483647` | window bounds |
| `common.scale` | `inv_sqrt_d` | `inv_sqrt_d` or a positive float |
| `common.return_softmax_lse` | `false` | whether to return lse |
| `common.phase_threshold` | `16` | decode/prefill boundary; optional |
| KV | dense | no `cu_seqlens`, no PA |

### 6.2 s2_curve

| Item | Current value | Notes |
|---|---|---|
| `enabled` | `true` | whether enabled |
| `batch` | `8` | batch size |
| `s1` | `[1, 3, 8, 16, 128, 256, 512, 2048, 8192]` | Q length list |
| `s2_aligned` | `[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]` | S2 alignment points |
| `s2_tail_sizes` | `[1, 16, 32, 64, 127]` | tail blocks appended after each alignment point |

### 6.3 batch_linearity

| Item | Current value | Notes |
|---|---|---|
| `enabled` | `true` | whether enabled |
| `batch_values` | `[1, 2, 4, 8, 16, 32, 64]` | batch values to scan |
| `shapes` | prefill `S1=4096,S2=6144`; decode `S1=1,S2=24576` | per-phase shapes |

### 6.4 s2_balance

| Item | Current value | Notes |
|---|---|---|
| `enabled` | `true` | whether enabled |
| `batch` | `16` | batch size |
| `S2` | `8192` | total KV length |
| `fill` | `0.5` | valid-length fill ratio |
| `distributions` | `[uniform, ramp, skew, outlier, alternating]` | per-batch length distribution |
| `shapes` | prefill `S1=4096`; decode `S1=1` | per-phase shapes |

An omitted `experiments` section is disabled; when present, `enabled` defaults to `true`. Unknown fields raise an error.

Device peaks are mapped from the model automatically: known models provide HBM peak and standard frequency; unknown models are not guessed and the corresponding metrics are left empty.

## 7. Known notes and constraints

- the result header matches `pytests` and `random_fuzz`; uncollected precision or profiling columns are left empty;
