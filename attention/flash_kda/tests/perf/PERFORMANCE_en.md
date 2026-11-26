# FlashKDA A5/A6 performance measurement

[中文](PERFORMANCE.md)

Both architectures read the same 25 profiling cases from `../assets/cases/flash_kda_perf_profiling.csv`, preserving inputs, seeds, fusion/cache switches, and training/inference semantics. Metadata uses native architecture capacity; transport preparation stays outside the measurement. Pass `--golden-mode Disable` when running pure TTK profiling to skip CPU golden (omit it to verify accuracy on selected cases).

## Two metrics

| Metric | Included | Excluded |
| --- | --- | --- |
| Main kernel | Profiler `Duration(us)` for the selected template's complete compute kernel | Metadata, auxiliary Cast/Transpose/Zero kernels, warmup, and initial compilation |
| End to end | Wall time for regenerating metadata, forward conversions/allocations, the public three-output entry including FP32-to-BF16 chunk-state conversion, and NPU synchronization | Input generation/H2D, warmup, initial compilation, and profiler overhead |

Sample the metrics separately on one exclusive device/stream. Defaults are 10 warmups, 50 samples, and three independent rounds; report the median of the three round means. `benchmark.py` records the reference commit as `interface_contract` in `environment.json`; `compare_performance.py` rejects missing or mismatched contracts.

## Recommended: one script for both metrics

Run from the repository root in each architecture's matching environment. `benchmark.py` defaults to `KDA_DUT=installed` with `CANNBOTDSL_NATIVE_BINARY_MODE=require`. Source measurements explicitly use `KDA_DUT=source`. The output directory must be outside the repository and must not already exist.

```bash
# A5 environment
export KDA_DUT=installed
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/perf/benchmark.py \
  --arch arch35 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a5-perf
python attention/flash_kda/tests/assets/impl/kda_runtime.py --check-native /tmp/kda-a5-perf/evidence

# A6 environment
export CANNBOTDSL_NPU_ARCH=dav-9201
export CANNBOTDSL_CACHE_DIR="$(mktemp -d /tmp/kda-a6-cache.XXXXXX)"
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/perf/benchmark.py \
  --arch arch92 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a6-perf
python attention/flash_kda/tests/assets/impl/kda_runtime.py --check-native /tmp/kda-a6-perf/evidence
```

Compare both result directories on one machine:

```bash
python attention/flash_kda/tests/perf/compare_performance.py \
  /tmp/kda-a5-perf /tmp/kda-a6-perf > /tmp/kda-performance-comparison.json
```

## TTK entry point

Run `flash_kda_perf_profiling.csv` directly with TTK (`torch.ops.cann_ops_transformer.flash_kda`):

```bash
KDA_ASSETS="attention/flash_kda/tests/assets"
KDA_PERF_TEMPLATE=auto python -m ttk e2e \
  -i "$KDA_ASSETS/cases/flash_kda_perf_profiling.csv" \
  --plugin "$KDA_ASSETS" --golden-mode Disable --dev 1 --pc 1 \
  --run 50 --warmup true -o "$KDA_OUT/kernel.csv"
python "$KDA_ASSETS/impl/kda_runtime.py" --check-native "$KDA_EVIDENCE_DIR"
```

`KDA_PERF_TEMPLATE` accepts `auto`, `staged`, or `persistent`. Prefer `benchmark.py` for three independent rounds and the two-metric A5/A6 comparison.
