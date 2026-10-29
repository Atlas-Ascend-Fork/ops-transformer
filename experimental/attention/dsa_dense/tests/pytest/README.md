# dsa_dense算子测试框架

## 功能说明
基于pytest测试框架，实现dsa_dense算子的功能验证：
- **CPU侧**：复现算子功能（DSA dense 反向的 loss/dq/dk/dw）生成 golden 数据
- **NPU侧**：通过 `torch_npu.dsa_dense` 直调获取实际数据
- **精度对比**：按浮点误差（mean(abs diff) <= 阈值）比较 CPU 与 NPU 结果

## 当前实现范围

### 参数限制
- 布局：TND（变长）
- 数据类型：BF16（query/key/queryIndex/keyIndex）、FLOAT（weights/max/sum/maxIndex/sumIndex）
- `sparse_mode` 仅支持 1（causal）；`D=192`、`DI=128`、`NI=N//2`

### 环境配置
1. 确认 TorchNPU 为最新版本
2. 参考[Attention融合算子Experimental使用说明](../../../Attention融合算子Experimental使用说明.md)激活 CANN 包和自定义算子包

## 文件结构
- test_run.sh                          # 执行脚本
- dsa_dense_golden.py              # cpu侧算子golden实现
- dsa_dense_process.py             # 调用算子获取npu输出
- dsa_dense_paramset.py            # 用例入参配置
- result_compare_method.py             # 精度对比
- test_dsa_dense_single.py         # pytest测试单用例运行主程序
- utils.py                             # 结果保存工具
- pytest.ini                           # 创建测试标记

## 使用方法

在 pytest 文件夹路径下执行：

1. 手动配置 `dsa_dense_paramset.py` 的参数
2. 执行指令：
```bash
bash test_run.sh single
```
支持自定义结果保存路径：
```bash
bash test_run.sh single -R "./result/dsa_result.xlsx"
```
