# FlashKDA A5/A6 统一性能测量

两侧直接读取 `../assets/cases/flash_kda_perf_profiling.csv` 中 25 条 profiling 用例，保持相同输入、seed、融合/缓存开关及训推语义。metadata 使用架构原生容量，公共传输准备在测量外；通过 TTK 跑纯性能剖析时请显式传入 `--golden-mode Disable` 跳过 CPU golden（不传时可按需对单条用例校验精度）。

## 两项指标

| 指标 | 纳入范围 | 排除范围 |
| --- | --- | --- |
| 主 kernel | profiler 中当前模板完整计算 kernel 的 `Duration(us)` | metadata、Cast/Transpose/Zero 等辅助核、预热、首次编译 |
| 端到端 | 每次重新生成 metadata、前向内部转换/分配、公开三项入口执行（包括公开 chunk_states 的 FP32→BF16 转换）和 NPU 同步的墙钟时间 | 输入构造/H2D、预热、首次编译、profiler 开销 |

两者单独采样。端到端不通过“主 kernel 加一次 metadata 时间”估算，也不包装或替换算子校验函数。每侧都在同一设备/stream 上串行执行，默认预热 10 次、采样 50 次、独立测量 3 轮，取三轮均值的中位数。两侧设备应独占，记录相应 SDK/wheel 指纹。

`benchmark.py` 在 `environment.json` 中记录固定参考提交作为 `interface_contract`；比较工具拒绝缺失或不匹配的接口标识。

## 推荐：同一脚本完成两种计时

以下命令分别在 A5、A6 匹配环境的仓库根目录执行；`benchmark.py` 默认 `KDA_DUT=installed`，三个接口来自专用 wheel 并要求 Native `require`；源码测量显式设置 `KDA_DUT=source`，独立归档。`SDK_TAG` 填实际 wheel/source 指纹，输出目录必须位于仓库外且不存在。

```bash
# A5 环境
export KDA_DUT=installed
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/perf/benchmark.py \
  --arch arch35 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a5-perf
python attention/flash_kda/tests/assets/impl/kda_runtime.py --check-native /tmp/kda-a5-perf/evidence

# A6 环境
export CANNBOTDSL_NPU_ARCH=dav-9201
export CANNBOTDSL_CACHE_DIR="$(mktemp -d /tmp/kda-a6-cache.XXXXXX)"
export CANNBOTDSL_NATIVE_BINARY_MODE=require
python attention/flash_kda/tests/perf/benchmark.py \
  --arch arch92 --sdk-tag "$SDK_TAG" --templates auto staged persistent \
  --output /tmp/kda-a6-perf
python attention/flash_kda/tests/assets/impl/kda_runtime.py --check-native /tmp/kda-a6-perf/evidence
```

只测自动选路时使用 `--templates auto`；定位关键 shape 时加 `--cases P22-001 P24-001`。

将两侧结果复制到同一机器后比较：

```bash
python attention/flash_kda/tests/perf/compare_performance.py \
  /tmp/kda-a5-perf /tmp/kda-a6-perf > /tmp/kda-performance-comparison.json
```

比较器拒绝输入哈希、属性、case/template、轮次或采样协议不一致的数据。逐例两项 A6/A5 比值均不超过 1 才通过；未达标退出码为 1。

## TTK 入口

通过 TTK 直接运行 `flash_kda_perf_profiling.csv`（公开算子 `torch.ops.cann_ops_transformer.flash_kda`）：

```bash
KDA_ASSETS="attention/flash_kda/tests/assets"
KDA_PERF_TEMPLATE=auto python -m ttk e2e \
  -i "$KDA_ASSETS/cases/flash_kda_perf_profiling.csv" \
  --plugin "$KDA_ASSETS" --golden-mode Disable --dev 1 --pc 1 \
  --run 50 --warmup true -o "$KDA_OUT/kernel.csv"
python "$KDA_ASSETS/impl/kda_runtime.py" --check-native "$KDA_EVIDENCE_DIR"
```

可将 `KDA_PERF_TEMPLATE` 改为 `staged` 或 `persistent`；`npu_preprocess` 在进入计时循环前生成对应模板的原生 metadata 并切换训推模式。独立三轮和双指标对比优先使用上面的 `benchmark.py`。
