# flash_attn Test Project

This directory contains three independently usable test tools:

| Tool | Directory | README | Purpose | Dispatch target |
|---|---|---|---|---|
| functional | `pytests/` | [README](pytests/README_en.md) | accuracy/functional verification, stage composition | `pytests/test_flash_attn.py` |
| random | `random_fuzz/` | [README](random_fuzz/README_en.md) | random fuzzing, long runs, failure collection and replay | `random_fuzz/run_random_fuzz.py` |
| perf | `perf_benchmark/` | [README](perf_benchmark/README_en.md) | msprof performance collection, backfill, plotting | `perf_benchmark/run_perf.py` |

## Unified Entry run.py

`run.py` is optional. It parses only the first argument as the tool name; the rest must follow it and are passed through unchanged:

```text
python3 run.py <functional|random|perf> [arguments...]
```

Each tool can also be launched directly; options, inputs/outputs, and boundaries are documented in its README.
