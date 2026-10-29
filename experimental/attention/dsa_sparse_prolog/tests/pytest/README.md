# dsa_sparse_prolog算子测试框架

## 功能说明
基于pytest测试框架，实现dsa_sparse_prolog算子的功能验证：
- **CPU侧**：fp64 真值 golden
- **NPU侧**：通过 `torch_npu.dsa_sparse_prolog` 直调获取实际数据，并运行 torch 拼接实现作为基线
- **精度对比**：op vs torch 基线 / op vs fp64 的 L2 相对误差，判据 `op_vs_fp64 <= torch_vs_fp64 * 1.5`（demote 模式追加 `op_vs_torch <= 2e-3`）

## 当前实现范围
- 布局：TND（`(T, He)`）与 `(B, S, He)`
- 数据类型：BF16 / FP16
- 权重布局：`nz_kn`（FRACTAL_NZ）、`nd_kn`、`nd_nk`

## 文件结构
- test_run.sh                          # 执行脚本
- dsa_sparse_prolog_golden.py      # cfg / RoPE / 权重 / torch golden / fp64 golden / rel_err
- dsa_sparse_prolog_process.py     # 调用算子获取 npu 输出 + torch 基线
- dsa_sparse_prolog_paramset.py    # 用例入参配置
- result_compare_method.py             # 精度对比
- test_dsa_sparse_prolog_single.py # pytest测试单用例运行主程序
- utils.py                             # 结果保存工具
- pytest.ini                           # 创建测试标记

## 使用方法
在 pytest 文件夹路径下执行：
```bash
bash test_run.sh single
```
支持自定义结果保存路径：
```bash
bash test_run.sh single -R "./result/dsa_prolog_result.xlsx"
```
