# pytests: flash_attn fixed-case precision tests

## 1. Overview

`pytests/` is the fixed-case precision test tool for flash_attn. It runs JSON-defined cases with pytest and performs input generation, reference computation, NPU execution, and precision verdicts for each case.

Main features:

- precision regression for fixed cases;
- filtering by case_id, tag, or skip list;
- two-way/three-way verdicts;
- saving inputs, reference outputs, and NPU outputs for diagnosis.

Not for:

- performance collection: use `perf_benchmark`;
- random tests and long runs: use `random_fuzz`;
- generating random cases: use `random_fuzz`.

## 2. Files

```text
pytests/
├── conftest.py            # options, case collection, test fixtures
├── pytest.ini             # marker registration
├── test_flash_attn.py     # pytest entry
└── cases/                 # JSON cases
```

## 3. Environment

Run from `attention/flash_attn/tests/pytests`:

```bash
cd attention/flash_attn/tests/pytests
source <CANN>/set_env.sh
```

A working TorchNPU environment is required. Graph mode also requires torchair.

## 4. Usage

### 4.1 Common usage and commands

#### 4.1.1 Run precision tests

```bash
python3 -m pytest -q --cases=cases/
```

`--cases` is required and may be repeated with multiple JSON files or directories. An empty selection is an error.

#### 4.1.2 Filter cases

```bash
python3 -m pytest -q --cases=cases/ --case-id=decode_BSND
```

- `--case-id`: comma-separated; without wildcards it matches a case_id substring, with wildcards it matches the full name, short name, or readable prefix;
- `--testset`: comma-separated tags;
- `--skip-cases`: a `.txt` file or comma-separated patterns; lines starting with `#` are comments.

### 4.2 Common options

| Option | Default | Allowed values / notes |
|---|---|---|
| `--cases` | none, required | JSON file or directory; repeatable |
| `--case-id` | no filter | comma-separated patterns; substring and wildcard matching |
| `--testset` | no filter | comma-separated tags |
| `--skip-cases` | no skip | `.txt` file or comma-separated patterns |
| `--stages` | `all` | `all` or a comma-separated combination of `gen`/`golden`/`run`/`compare`; see 4.3 |
| `--golden` | `device=npu,compute_dtype=fp32,p_dtype=out_dtype` | comma-separated `key=value`; fields and defaults are listed in 4.4 |
| `--compare-mode` | `two_way` | `two_way` or `three_way` |
| `--out-dir` | `tests/out/functional-<timestamp>/` | any writable directory |
| `--save-data` | off | boolean; saves `inputs.pt`, `golden.pt`, `npu_out.pt` |
| `--mem-guard-gb` | `32` | float >= 0; 0 disables only the host guard, the HBM x 0.9 guard remains active |
| `--graph` | off | boolean; requires torchair |
| `--device-id` | `0` | integer >= 0 |

Resource and execution control:

- `--mem-guard-gb=16`: cases whose estimated host peak exceeds 16 GiB are marked `skip`;
- `--graph`: graph execution; the first run includes graph build/compile time; missing torchair raises an error instead of falling back to eager.

### 4.3 `--stages` combinations

| Combination | Supported | Required data | Notes |
|---|---|---|---|
| empty | ❌ Unsupported | - | no effective stage; not recommended |
| `gen` | ✅ Supported | no saved data | generate inputs only |
| `golden` | ✅ Supported | `inputs.pt` | compute the reference only |
| `run` | ✅ Supported | `inputs.pt` | execute the operator only |
| `compare` | ❌ Unsupported | - | missing `run`; `npu_out.pt` is not loaded |
| `gen,golden` | ✅ Supported | no saved data | generate inputs and compute the reference |
| `gen,run` | ✅ Supported | no saved data | generate inputs and execute |
| `gen,compare` | ❌ Unsupported | - | missing reference and `run` |
| `golden,run` | ✅ Supported | `inputs.pt` | compute the reference and execute |
| `golden,compare` | ❌ Unsupported | - | missing `run` |
| `run,compare` | ✅ Supported | `inputs.pt`, `golden.pt` | execute and compare using saved reference |
| `gen,golden,run` | ✅ Supported | no saved data | inputs, reference, and execution ready; not compared |
| `gen,golden,compare` | ❌ Unsupported | - | missing `run` |
| `gen,run,compare` | ✅ Supported | `golden.pt` | generate inputs, execute, and compare |
| `golden,run,compare` | ✅ Supported | `inputs.pt` | compute reference, execute, and compare |
| `gen,golden,run,compare` | ✅ Supported | no saved data | full flow |

`--stages` defaults to `all`. Without `gen`, inputs are loaded from `<out-dir>/<case_id>/inputs.pt`. Without `golden` but with `compare`, the reference is loaded from `golden.pt`. Use `--save-data` when reusing stage data across commands.

### 4.4 `--golden` configuration

`--golden` selects the reference computation. By default the reference runs on NPU with fp32 accumulation and casts P to the V dtype. For a high-precision reference, use `device=cpu,compute_dtype=fp64,p_dtype=fp32`.

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

Results are written to `<out-dir>/results.csv`, one row per case:

- `pass`: precision passed;
- `fail`: precision failed;
- `skip`: rejected by the operator or skipped by a guard;
- `error`: execution or comparison error.

With `--save-data`, `inputs.pt`, `golden.pt`, and `npu_out.pt` are saved under `<out-dir>/<case_id>/`; nothing is saved by default.

Read `case_id` from `results.csv`. It is stable for the same `params`, independent of machine, path, and time.

## 6. Precision case construction

Case format (two cases):

```json
{
  "schema": "case/v1",
  "count": 2,
  "cases": [
    {
      "op": "flash_attn",
      "params": {
        "dtype": "fp16",
        "layout_q": "BNSD",
        "B": 2,
        "N1": 8,
        "S1": 256,
        "S2": 256,
        "D": 128,
        "mask_mode": 0
      },
      "tags": ["ci"]
    },
    {
      "op": "flash_attn",
      "params": {
        "dtype": "bf16",
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "layout_out": "BSND",
        "B": 1,
        "N1": 8,
        "N2": 2,
        "S1": 145,
        "S2": 257,
        "D": 64,
        "DV": 64,
        "mask_mode": 3,
        "return_softmax_lse": true,
        "seed": 13
      },
      "tags": ["func_rdv"]
    }
  ]
}
```

The top level may also be a case array directly. Each element of `cases` is one case; a single case object cannot be the top level of a case file.

File fields:

| Field | Meaning | Required | When omitted |
|---|---|---|---|
| `schema` | case file format version | no | treated as `case/v1`; any other value is an error |
| `count` | number of `cases` | no | not checked; when present it must equal the array length |
| `cases` | array of single cases | yes | file validation fails |
| `op` | operator name of a single case | no | defaults to `flash_attn` |
| `params` | operator parameter object of a single case | yes | file validation fails |
| `tags` | pytest markers | no | empty by default |

`params` fields are listed below; parameter meanings and value ranges follow the operator specification:

| Parameter | Meaning | Required | When omitted |
|---|---|---|---|
| `dtype` | data type | no | defaults to `fp16` |
| `N1` | Q head count | yes | normalization fails |
| `N2` | KV head count | no | defaults to `N1` |
| `D` | Q/K head size | yes | execution setup fails |
| `DV` | V head size | no | defaults to `D` |
| `B` | batch size | required when not TND | derived from `cu_seqlens` for TND; dense cases cannot be constructed |
| `S1` | Q sequence length | required when not TND | derived from `cu_seqlens_q` for TND; dense cases cannot be constructed |
| `S2` | KV sequence length | no | defaults to `S1`; for TND it is derived from `cu_seqlens_kv` |
| `layout_q` | Q layout | no | defaults to `BNSD` |
| `layout_kv` | KV layout | no | defaults to `layout_q` |
| `layout_out` | output layout | no | defaults to `layout_q` |
| `mask_mode` | mask type | no | defaults to `0` |
| `scale` | attention scale | no | defaults to `1/sqrt(D)` |
| `win_left`/`win_right` | window bounds | no | default to INT_MAX |
| `cu_seqlens_q` | TND Q segment lengths | required when `layout_q=TND` | ignored when not TND; missing for TND is an error |
| `cu_seqlens_kv` | TND KV segment lengths | no | defaults to `cu_seqlens_q`; ignored when not TND |
| `seqused_q`/`seqused_kv` | valid lengths | no | derived from segment lengths for TND; PA `seqused_kv` defaults to `[S2]*B` |
| `max_seqlen_q`/`max_seqlen_kv` | maximum lengths | no | derived from `seqused`, `cu_seqlens`, or `S1`/`S2` |
| `return_softmax_lse` | whether to return lse | no | defaults to `false` |
| `sinks` | whether to generate sinks | no | not generated by default |
| `block_size` | PA page size | no | defaults to `128` for PA |
| `seed` | input random seed | no | derived from a hash of `params` |

`tags` provides pytest markers; `params.seed` fixes the input seed.

## 7. Known notes and constraints

- precision only; no performance collection;
- PA K/V are expanded to dense on the reference side; this is included in memory estimation, and oversized cases are marked `skip`;
- explicit `scale<=0` is rejected as an invalid input and marked `skip`; operator-reported unsupported/tiling failures are also `skip`, not `fail`;
- unrecoverable device/queue errors abort the run immediately and keep the results already written;
- `--stages` is not an arbitrary combination: `golden` needs saved inputs, and `compare` needs the current run result;
- xlsx/zip/CSV inputs are not supported; cases use JSON only, and CSV is for results only;
- use `random_fuzz` for random cases; this tool executes existing JSON only.
