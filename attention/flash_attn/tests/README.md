# flash_attn测试工程

本目录包含三个可独立使用的测试工具：

| 工具 | 目录 | README | 用途 | 透传目标 |
|---|---|---|---|---|
| functional | `pytests/` | [README](pytests/README.md) | 精度/功能验证、stage组合 | `pytests/test_flash_attn.py` |
| random | `random_fuzz/` | [README](random_fuzz/README.md) | 随机fuzz、长跑、失败收集与重放 | `random_fuzz/run_random_fuzz.py` |
| perf | `perf_benchmark/` | [README](perf_benchmark/README.md) | msprof性能采集、回填和绘图 | `perf_benchmark/run_perf.py` |

## 统一入口run.py

`run.py`可选，只解析第一个参数为工具名，其余参数须放在工具名之后并原样透传：

```text
python3 run.py <functional|random|perf> [参数...]
```

各工具也可直接启动；参数、输入输出和边界见对应README。
