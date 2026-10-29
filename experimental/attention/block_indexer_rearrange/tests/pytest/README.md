# block_indexer_rearrange算子测试框架

## 功能说明
基于pytest测试框架，实现block_indexer_rearrange算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过TorchNPU进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精确对比验证算子功能

## 当前实现范围

### 参数限制
- **数据格式：** TND（变长）
- **数据类型：** INT32（索引）、INT64（序列长度）

### 环境配置
#### 前置要求
1. 确认TorchNPU为最新版本
2. 参考[Attention融合算子Experimental使用说明](../../../Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包

## 文件结构
- test_run.sh                                      # 执行脚本
- block_indexer_rearrange_golden.py            # cpu侧算子golden实现
- result_compare_method.py                         # cpu golden与npu输出精度对比
- block_indexer_rearrange_process.py           # 调用算子获取npu输出
- block_indexer_rearrange_paramset.py          # 用例入参配置
- test_block_indexer_rearrange_single.py       # pytest测试单用例运行主程序
- utils.py                                         # 结果保存工具
- pytest.ini                                       # 创建测试标记

## 使用方法

在pytest文件夹路径下执行：

1. 手动配置`block_indexer_rearrange_paramset.py`的参数
2. 执行指令：
```bash
bash test_run.sh single
```

支持自定义结果保存路径：
```bash
bash test_run.sh single -R "./result/bir_result.xlsx"
```
