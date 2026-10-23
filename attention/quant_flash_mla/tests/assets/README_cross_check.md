# QuantFlashMla：三方 cross_check 比对

FP8 全量化下，CPU golden 与 kernel 的 P 量化路径不同（kernel 按分核出的
s2 分段独立取 running max 并量化 P）。直接用"NPU 对 golden"的两方判据，会把
golden 自身的量化噪声当成误差，产生假失败。

三方比对改为以一份高精度 bench 为共同基准：

| 角色 | 来源 | `CrossCheckComparison` 参数位 |
| --- | --- | --- |
| target | 被测 quant_flash_mla 算子输出 | `output` |
| bench | 新增 fp64 CPU 参考（全局 running max） | `golden`（共同基准） |
| third_party | 分段量化 CPU 参考 | `third_party` |

判据是"NPU 相对 bench 的误差 / CPU 参考相对 bench 的误差"，即 NPU 的误差不超过
参考噪声的若干倍。

## 调用链

```text
customize_inputs  →  inputs.generate_qfmla_inputs()      固定种子生成算子输入
        │
        ├─ 算子执行                          → outputs = [atten(bf16), lse(fp32)?]
        │
golden  →  impl/golden.cpu_qfmla()
             ├─ generate_data()  同种子重放，得到与算子逐位相同的 q/k/v/descale
             ├─ fp8_fullquant_mla_golden()            → 两方 golden（全局量化）
             └─ _cache_bench()
                  ├─ fp8_fullquant_mla_bench()        → _BENCH_CACHE[case]
                  └─ fp8_fullquant_mla_sectioned_golden() → _REF_CACHE[case]
        │
compare →  impl/compare.compare()  ← spec 按 compare_context.testcase_name 取两个缓存
```

`spec.compare` 只在 `cross_check_enabled()` 为真时传 `bench_outputs` /
`third_party_outputs`；否则走原有两方比较，不生成任何额外参考。

## bench 口径

- 计算精度：fp64 累加、fp64 softmax，不复现 fp32 累加误差
- 输入：反量化后的 q/k/v（只乘 descale），**不重新量化**
- 保留算子语义的 P 固定 scale cast：FP8 为 `p*448 → e4m3 → fp32`
- running max 为**全局**（整个 KV 一次），**不复现 metadata 分段**
- 输出：fp32，按用例的输出布局（BSND/BNSD/TND）转换后缓存

## third_party（分段参考）口径

`fp8_fullquant_mla_sectioned_golden` 复刻 kernel 的分段量化行为：

- 切段：`m_base = 64`，`s2_base = 256`，**每个 s2 基本块单独成段**
- 段内：独立 running max，P 按上述固定 scale cast，累加 `(o, sum, m)`
- 段间：按 FD 语义合并 `o = o*exp(m_old-m_new) + o_seg*exp(m_seg-m_new)`
- 输出：转 bf16 并与算子输出同形状

**故意不依赖 metadata 分核结果。** kernel 的真实分段是"若干连续基本块在分核后
的并集"，按单块切段是其粒度最细的上界：段数只会更多，参考的量化噪声只会不小于
target，因此 ratio 不会被过干净的参考抬高。代价是它与 kernel 不逐元素一致。

metadata 分核版本（`get_qfmla_section_info`）保留在 `qfmla_golden.py` 中，由
`ENABLE_SECTION_GOLDEN` 控制，默认关闭。

## 判定方法

`compare._run_cross_check` 逐输出通道处理，通常为 atten（bf16）+ lse（fp32）：

1. 三个数组统一升 fp32，形状必须一致；
2. 大值域 `|bench| >= small_value`：
   - `mare = max|target-bench| / max|third-bench|`
   - `mere = mean|target-bench| / mean|third-bench|`
   - `rmse = sqrt(mean((target-bench)^2)) / sqrt(mean((third-bench)^2))`
3. 小值域 `|bench| < small_value`：统计 `|x-bench| > small_value_atol` 的点数，
   `count_target / count_third <= small_value_ratio`；
4. NaN/Inf 走框架内置的三方一致性判定；
5. 除法分母保护、分子分母下限由 TTK `CrossCheckComparison` 内部实现。

三项比值均不超过上表阈值即 PASS。

## 阈值

标准 `cross_check`、等级 `L0`、`mare/mere/rmse = 10/2/2`：

| 输出 dtype | small_value | small_value_atol | small_value_ratio |
| --- | --- | --- | --- |
| float16 | `2^-10` | `2^-16` | 2 |
| bfloat16 | `2^-10` | `2^-16` | 2 |
| float32（lse） | `2^-14` | `2^-30` | 2 |
