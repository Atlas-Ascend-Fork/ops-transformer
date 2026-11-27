# random_fuzz: random flash_attn tests

## 1. Overview

`random_fuzz/` is the main random test tool for flash_attn. It generates legal random cases in batches, runs functional/precision tests, and continuously writes results and failing cases.

Main features:

- run random cases within the operator-supported scope; see Section 6, "Random case scope and configuration";
- specify the number of test cases with `--max-cases` or the test duration with `--max-seconds`;
- precision failure collection and replay;
- sampling of large cases and boundary lengths.

It also provides standalone random case JSON generation for inspecting the input distribution or later replay.

Not for:

- performance collection: use `perf_benchmark`;
- fixed-case precision regression: use `functional`;
- saving inputs/golden/npu_out: not provided by random_fuzz.

## 2. Files

```text
random_fuzz/
├── gen_random_cases.py    # random case generator
├── run_random_fuzz.py     # random test entry
└── README.md
```

## 3. Environment

Run from `attention/flash_attn/tests`:

```bash
cd attention/flash_attn/tests
source <CANN>/set_env.sh
```

A working TorchNPU environment is required. Graph mode also requires torchair.

## 4. Usage

### 4.1 Run random tests

```bash
cd attention/flash_attn/tests
python3 random_fuzz/run_random_fuzz.py --max-cases=1000 --out-dir=out/random-run1
```

Adjust the case count, time budget, and failure limit with the options below. Append `--mem-guard-gb=16` or `--graph` when needed.

### 4.2 Generate random cases

```bash
cd attention/flash_attn/tests
python3 random_fuzz/gen_random_cases.py --seed 7 --out random_fuzz/cases/random.json
```

The case count is controlled by `GEN_CASES`. This JSON is for inspecting the input distribution or replay; it is not required to run random tests.

### 4.3 Common options

| Option | Default | Allowed values / notes |
|---|---|---|
| `--out-dir` | `tests/out/random-<timestamp>/` | any writable directory |
| `--device-id` | `0` | integer >= 0 |
| `--seed` | `7` | integer; the same seed and generation config reproduce case shapes |
| `--max-cases` | `0` | integer >= 0; 0 = unlimited, but `--max-seconds>0` is required |
| `--max-seconds` | `0` | float >= 0; 0 = unlimited |
| `--max-failures` | `0` | integer >= 0; 0 = unlimited |
| `--mem-guard-gb` | `32` | float >= 0; 0 disables only the host guard, the HBM x 0.9 guard remains active |
| `--graph` | off | boolean; requires torchair |
| `--golden` | `device=npu,compute_dtype=fp32,p_dtype=out_dtype` | comma-separated `key=value`; fields and values are listed below |
| `--compare-mode` | `two_way` | `two_way` or `three_way` |

`--golden` fields:

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

When `--max-cases=0`, `--max-seconds>0` is required; otherwise the command exits with an error to avoid an infinite run.

- `--mem-guard-gb=16`: cases whose estimated host peak exceeds 16 GiB are skipped during both generation and execution;
- `--graph`: graph execution; the first run includes graph build/compile time; missing torchair raises an error instead of falling back to eager.

## 5. Output

A checkpoint is written after each batch:

```text
<out-dir>/
├── results.csv
└── failures.json   # only when there are failures
```

Common statuses:

- `pass`: precision passed;
- `fail`: precision failed;
- `error`: execution or comparison error;
- `skip`: rejected by the operator or skipped by a guard;
- `fatal`: unrecoverable device/queue error.

`failures.json` contains `op`, `tags`, full `params`, `case_id`, `status`, `detail`, and precision metrics.

### 5.1 Long runs and batch size

The runner generates and executes in batches; the default is `GEN_CASES_PER_BATCH=1000` cases per batch. After each batch, cumulative results are rewritten to `results.csv` and `failures.json`.

- interruption after a batch: completed batches are on disk;
- interruption mid-batch: cases already executed in the current batch but not yet written are lost; a time-budget or fatal exit makes a best-effort checkpoint before exiting;
- resume is not supported; a restart counts from zero and rewrites `results.csv`; an old `failures.json` is not deleted automatically, so use a new `--out-dir` to keep old results.

Change the batch size at the top of `run_random_fuzz.py`; there is no command-line option:

```python
GEN_CASES_PER_BATCH = 1000
```

A smaller batch reduces loss on interruption but rewrites the CSV more often; a larger batch does the opposite. `--max-cases` only limits the total and does not change the batch size.

### 5.2 Reproduction

Replay failures with the functional entry:

```bash
python3 -m pytest -q pytests/test_flash_attn.py --cases=<out-dir>/failures.json
```

- use the same `--seed` and generation config to reproduce case shapes;
- `params.seed` in `failures.json` reproduces a single input.

## 6. Random case scope and configuration

The generation scope, current values, and corresponding variables for random cases are listed below:

| Item | Current values / range | Variables |
|---|---|---|
| dtype | `fp16`, `bf16` | `DTYPES` |
| Q/OUT layouts | `BNSD`, `BSND`, `TND`, and mixed Q/OUT combinations | `Q_OUT_LAYOUTS` |
| KV layouts | same as Q (dense/TND), `PA_BBND`, `PA_BNBD`, `PA_NZ` | `KV_LAYOUTS`, `PA_LAYOUTS` |
| mask | `0`, `3`, `4`; window is supported for `4` | `MASK_MODES` |
| D/DV | `(64,64)`, `(72,72)`, `(128,128)`, `(192,128)`, `(256,256)` | `SUPPORTED_DV` |
| seqlen scenarios | `S1_varlen`, `S2_trunc`, `S3_varlen_trunc`, `S4_exact`, `S5_empty`, `S6_pa_decode`, `S7_tnd_pa_varlen` | `SEQ_SCENES` |
| lengths/batch/heads | Q/KV `0..1M`, B `1..512`, N2 `1..16`, G `1..32`, decode S1 `1..16` | `Q_SEQ_RANGE`, `KV_SEQ_RANGE`, `BATCH_RANGE`, `N2_RANGE`, `G_RANGE`, `DECODE_S1_RANGE` |
| seqlen sampling | 20% boundary + 80% log-uniform; pad <= 256; total tokens <= 1M | `SEQ_BOUNDARY_RATE`, `NORMAL_LOG_MIN`, `PAD_MAX`, `MAX_TOTAL_TOKENS` |
| scale | default, inv_sqrt_d, `(0,1]`, `(1,FP32_MAX)` | `SCALE_STRATEGIES` |
| window | both_inf, left_inf, right_inf, boundary, uniform_range, overflow_left/right/both | `WINDOW_STRATEGIES`, `WINDOW_CANDIDATES` |
| PA block size | `16..1024`, step 16 | `BLOCK_SIZE_RANGE` |
| optional inputs | `return_softmax_lse`, `sinks`, explicit `max_seqlen_q/kv`, `head_dim_v` | `RETURN_SOFTMAX_LSE`, `SINKS`, `EMIT_MAX_SEQLEN_Q`, `EMIT_MAX_SEQLEN_KV`, `EMIT_HEAD_DIM_V` |
| large cases | `>=128K`, 1% | `LARGE_SEQ_THRESHOLD`, `LARGE_CASE_RATE` |
| retries/memory guard | max 100 attempts per case, safety factor 1.2 | `DEFAULT_ATTEMPTS_PER_CASE`, `GEN_SAFETY_FACTOR` |
| case count | default 1000 | `GEN_CASES` |
| execution batch size | default 1000 | `GEN_CASES_PER_BATCH` |

All variables are at the top of `gen_random_cases.py` except `GEN_CASES_PER_BATCH`, which is at the top of `run_random_fuzz.py`. Execution is eager by default; `--graph` switches to graph mode.

## 7. Known notes and constraints

- random_fuzz performs functional/precision tests only; it does not collect performance;
- it does not save `inputs/golden/npu_out` and has no `--save-data`;
- it does not provide `--case-id`/`--testset`/`--skip-cases` because cases are random; to filter, write cases to JSON first and use the functional entry;
- device fatal errors are recorded as `fatal`; the run stops only after 20 consecutive fatal errors;
- the generation flow is: select scenario -> sample parameters -> legality/memory checks -> retry invalid candidates -> execute and write results; when the attempt limit is reached, existing cases are returned and the shortfall is reported;
- generation may return fewer cases than requested and does not abort the whole run for a single failed candidate.
