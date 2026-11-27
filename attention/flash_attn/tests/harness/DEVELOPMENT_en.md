# flash_attn Test Framework Development Guide

Read this before modifying shared harness, golden, memory, compare, or paged paths.

## 0. Audience and Scope

This document is for framework maintainers and operator-adapter developers.

- User commands, options, and usage examples are in the frontend READMEs: `../pytests/README_en.md`, `../random_fuzz/README_en.md`, `../perf_benchmark/README_en.md`;
- This document follows the order "execution flow -> layered structure (framework/operator) -> change entry points -> conventions -> compatibility" and explains boundaries, contracts, extension points, and change rules;
- Minimal local check: `python3 -m pytest harness/tests harness/ops/flash_attn/tests perf_benchmark/tests -q`; run `pre-commit run --files <changed files>` before submitting;
- It does not cover operator math derivations, kernel performance tuning, or CI infrastructure.

## 1. Execution Flow

### 1.1 Entry and stages

```text
run.py / pytest frontend / random_fuzz / perf_benchmark
└─ harness/runner.py
   ├─ harness/ops/flash_attn/package.py
   ├─ harness/guards.py
   ├─ harness/compare.py
   ├─ harness/perf.py
   └─ harness/results.py
```

Stages are fixed:

```text
gen -> golden -> run -> compare
```

perf is a mode of `run`, not a separate stage; only `perf_benchmark` uses it.

### 1.2 Stage responsibilities

| Stage | Framework | Operator | Output |
|---|---|---|---|
| `gen` | scheduling, memory/legality guards, input digest, optional persistence | `build_inputs` | `inputs`, digest |
| `golden` | scheduling, golden config/fingerprint, optional persistence | `golden` | `golden` |
| `run` | scheduling, execution mode, optional persistence, perf collection | `execute` | `npu_out` or perf rows |
| `compare` | scheduling, result recording and writing | `compare` | metrics, passed |

- `compare` needs `golden` and the output of the current `run`. If golden was not computed, only saved `golden.pt` is loaded; `npu_out.pt` is not loaded.
- `inputs.pt`, `golden.pt`, and `npu_out.pt` are persisted only with `--save-data`; staged execution depends on these files.

### 1.3 Frontend entry and shared options

- `run.py` maps `functional` to `pytests/test_flash_attn.py`, `perf` to `perf_benchmark/run_perf.py`, and `random` to `random_fuzz/run_random_fuzz.py`; it only parses the mode and passes arguments through.
- All three tools execute through `harness.Runner`.
- `cli_common.py` defines shared options; `--golden` and `--compare-mode` defaults/choices come from `DEFAULT_GOLDEN_SPEC` and `COMPARE_MODES`.

## 2. Layered Structure

### 2.1 Framework layer

`runner.py` is the framework scheduling core. It owns stage order, exception classification, and result records for `gen -> golden -> run -> compare`; during `run`, it enters performance mode according to the `perf` setting and calls `perf.py` for performance collection. `runner.py`, `files.py`, and `guards.py` are cross-stage files, so stage tables list them again for each usage.

#### 2.1.1 Stages

##### gen

| File | Purpose | Responsibility |
|---|---|---|
| `case.py`, `loader.py` | input preparation | parse `case/v1`, build `Case`, compute stable `case_id`, filter by the requested conditions |
| `runner.py` | stage scheduling | call operator `build_inputs`, handle exceptions and result records |
| `files.py` | input persistence | compute `inputs_digest`; with `--save-data`, save `inputs.pt`/`inputs_meta.json` |
| `guards.py` | pre-generation guards | call `legalize`, estimate host/HBM peaks, decide `skip` |

##### golden

| File | Purpose | Responsibility |
|---|---|---|
| `runner.py` | stage scheduling | call operator `golden`, handle exceptions and result records |
| `golden_spec.py` | reference configuration | parse device/compute_dtype/p_dtype fields, produce fingerprint |
| `files.py` | reference data access | load `inputs.pt` when needed; with `--save-data`, save `golden.pt`/`golden.json` |

##### run

| File | Purpose | Responsibility |
|---|---|---|
| `runner.py` | execution scheduling | select eager/graph/perf mode; error out when torchair is missing in graph mode |
| `files.py` | output persistence | save `npu_out.pt` with `--save-data` |
| `perf.py` | perf-mode collection | L2 flush, profiler start/read, MFU/MBU, roofline theory time |

##### compare

| File | Purpose | Responsibility |
|---|---|---|
| `runner.py` | stage scheduling | invoke accuracy verdict, classify `pass/fail/skip/fatal`, record metrics |
| `compare.py` | accuracy verdict | two-way/three-way comparison, dtype tolerances, failure-sample statistics |
| `files.py` | reference data loading | load saved `golden.pt` when golden was not computed |
| `results.py` | result writing | define column order, flatten operator metrics, write the single `results.csv` |

#### 2.1.2 Performance data collection

| File | Purpose | Responsibility |
|---|---|---|
| `perf.py` | performance collection | L2 flush, msprof profiler, task-time parsing, MFU/MBU, and roofline theory time |
| `device_spec.py` | device peaks | maps device model to HBM peak and standard frequency; `perf_worker` reads the peak, and `backfill` fills standard frequency/peak after msprof; not part of the main stages |
| `perf_benchmark/` | performance frontend | worker, backfill, plotting, and performance case generation |

Performance chain: `perf_benchmark/run_perf.py` → `perf_worker.py` (`Runner(perf=True)`) → msprof → `backfill.py` → `plot_bench.py`; `device_spec.py` provides peaks to the worker and backfill.

### 2.2 Operator adaptation layer

#### 2.2.1 Contract and common

| File | Purpose | Responsibility |
|---|---|---|
| `package.py` | operator contract | expose `legalize`/`normalize`/`build_inputs`/`golden`/`execute`/`compare`/`perf_context`/`theoretical_workload`; the framework calls the operator only through it |
| `params.py` | parameter normalization | fill defaults, handle dtype aliases, derive TND/PA values, drop empty `seqused`/`cu_seqlens`, and keep parameters executable |
| `memory.py` | memory estimation | `tensor_footprint` counts tensor usage; `golden_footprint`/`estimate` combine stage peaks |

#### 2.2.2 gen

| File | Purpose | Responsibility |
|---|---|---|
| `inputs.py` | input generation | generate q/k/v/block_table/sinks by layout/PA; support range sampling, TND variable length, and paged page layouts |

#### 2.2.3 golden

| File | Purpose | Responsibility |
|---|---|---|
| `golden_truth.py` | CPU/GPU reference | row-blocked reference implementation; handle layout conversion, PA densification, and paged loading; provide `golden_peak` |
| `golden_npu.py` | NPU reference | run the same reference logic on device for the default `device=npu` |
| `policy.py` | P storage policy | resolve `p_block`, `softmax_mode`, and `softmax_block_size`; decide whether P is blocked/stored |

#### 2.2.4 run

| File | Purpose | Responsibility |
|---|---|---|
| `executor.py` | NPU invocation | call the operator in eager/graph mode and assemble metadata and kernel arguments |
| `layout.py` | layout/PA conversion | BNSD/BSND/TND conversion, PA page loading, and `PagedKVLoader`; does not know `p_dtype` |

`compare` has no separate operator file: `package.compare` dispatches to the generic comparison logic. Methods in `package.py` are called by `runner.py` at the corresponding stages.

### 2.3 Boundary constraints

**Between the three tools**

- `functional`, `random`, and `perf` must not import each other; each provides its own entry point.
- `functional` is pytest-driven only; `random` and `perf` must not enter the pytest process.
- None of them may duplicate `gen/golden/run/compare` scheduling.

**Between the tools and harness**

- All three must execute through `harness.Runner` and interact only through public interfaces such as `Runner`, `PACKAGE`, and `results`.
- `perf` must run msprof backfill and plotting outside Runner; `run.py` may only dispatch and must not implement business logic.
- `harness` must not import any frontend; `perf_benchmark` must not be imported by the pytest frontend or `runner.py`.

**Between harness and the operator layer**

- The framework must not import operator internals; it reaches the operator only through `PACKAGE`. The operator must not import `runner.py` in reverse.
- The framework must not contain flash_attn-specific fields or branches. Operator metrics are returned by the operator; the framework only flattens and writes them.

**Other**

- Before changing stage behavior, decide whether it belongs to the framework or the operator.
- pytest case collection, markers, and fixtures must stay in the pytest frontend.

## 3. Change Entry Points

### 3.1 Common

| What to change | Where |
|---|---|
| Add a case field | `ops/flash_attn/params.py` + `package.py` `legalize` |
| Tensor footprint | `ops/flash_attn/memory.py` |
| Memory/legality guards | `harness/guards.py` |
| golden/compare memory behavior | `memory.estimate`, `golden_peak`, guard tests, and README/development docs |

### 3.2 gen

| What to change | Where |
|---|---|
| Input data/distribution | `ops/flash_attn/inputs.py` |

### 3.3 golden

| What to change | Where |
|---|---|
| golden math | `ops/flash_attn/golden_truth.py` (CPU/reference), `golden_npu.py` (NPU) |
| golden peak | `golden_truth.golden_peak` |
| P precision/block policy | `ops/flash_attn/policy.py` + `harness/golden_spec.py` |

### 3.4 run

| What to change | Where |
|---|---|
| NPU invocation | `ops/flash_attn/executor.py` |
| PA/page loading | `ops/flash_attn/layout.py` |

### 3.5 compare

| What to change | Where |
|---|---|
| Accuracy verdict | `harness/compare.py` |
| Result columns/table | `harness/results.py`; sync CSV read/write and tests when adding or removing a column |

### 3.6 Performance

| What to change | Where |
|---|---|
| Performance/frequency/MFU | `harness/perf.py` |
| Device peak/standard frequency (perf only) | `harness/device_spec.py` |
| Offline performance analysis | `perf_benchmark/` |


### 3.7 Operator adaptation order

1. `package.py`: implement contract methods and `normalize/legalize`;
2. `params.py`: fill defaults, derive values, and validate parameters;
3. `inputs.py`: implement input generation;
4. `golden_truth.py`/`golden_npu.py`: implement references; `policy.py`: implement P storage policy;
5. `executor.py`/`layout.py`: implement execution and layout/PA conversion;
6. `memory.py`: implement `tensor_footprint`/`estimate`;
7. add unit tests and update README/development docs.

## 4. Conventions

### 4.1 Memory Estimation and Guards

- Estimation happens before `gen/golden` and must be pure computation:
  - do not create NPU tensors;
  - do not execute golden;
  - do not generate large `block_table` tensors.
- `tensor_footprint` counts only input/output/aux/lse tensors.
- `golden_footprint` counts only the golden-stage peak, including fp32 intermediates, K/V/k_t, chunk scratch, and paged block buffers.
- `persist` must equal this run's `save_data` and must not be hard-coded; otherwise the guard may falsely `skip` or underestimate memory.

### 4.2 Golden Accuracy

`GoldenSpec` fields and values are documented in the `--golden` table of `../pytests/README_en.md`.

- `p_dtype`, `p_block`, and `kv_loader` are independent parameters: `p_dtype` controls P storage precision, `p_block` controls the P block size, and `kv_loader` controls KV loading; setting one does not change the other two.
- Golden accuracy must not regress; the kernel is not required to match golden bit-for-bit and uses dtype tolerances.
- Floating-point accumulation stays fp32/fp64; do not switch to fp16 accumulation to save memory.

### 4.3 Results and Errors

- There is one `results.csv`; one row per kernel execution.
- `status`: `pass` / `fail` / `error` / `skip` / `fatal` / `perf`; `perf` only appears in performance mode without `--verify`.
- Device/queue fatal errors are `fatal`, not accuracy failures.
- `failures.json` from `random_fuzz` keeps `op`, `case_id`, `tags`, `params`, `status`, `detail`, and full `metrics` for replay.
- `results.csv` flattens only `_passed`, `_fail_ratio`, and `_max_abs` per accuracy metric; full comparison values such as `fail_cnt`, `samples`, and `max_rel` exist only in the `metrics` of `failures.json`.
- The core and operator-extension columns in `results.csv` are defined by `harness/results.py`; new profiling columns must be appended without changing existing semantics.
- Do not add CSV columns for temporary/debug information.

### 4.4 Implementation Conventions

- Before a memory-related change, report the peak increase and confirm it; do not implement first and document later.
- After changing memory estimation, rerun the baseline and confirm the skip-set change.
- Preallocate large outputs in the final dtype or write them in chunks; avoid the “full fp32 → stack/cat → cast” peak.
- Release fp32 intermediates such as `out_b` after appending; otherwise `stack` still holds host/device memory.
- Do not call `.item()` on every compare chunk; it causes frequent device synchronization.
- Converting large slices of non-contiguous NPU tensors to float may trigger AsStrided issues; call `contiguous` when needed.
- Model keys in `device_spec.py` must use actual device names; unknown models must not have inferred peaks.

## 5. Compatibility and Versions

- Case JSON uses `case/v1`; a `NORM_VERSION` change changes stable `case_id` and requires re-confirming the baseline;
- New `GoldenSpec` fields must update both `fingerprint()` and `describe()`; otherwise different configurations are treated as identical and an old `golden.pt` may be reused;
- `run.py` mode names and argument pass-through are stable interfaces; new frontends must not change existing modes;
- When changing the result schema, case schema, or default paths, update the user READMEs, development docs, and tests.
