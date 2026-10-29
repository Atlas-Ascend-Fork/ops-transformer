# flash_attention算子测试框架

## 功能说明
基于pytest测试框架，实现flash_attention算子（前向）的功能验证：
- **CPU侧**：变长 TND 因果/非因果注意力 golden（torch）
- **NPU侧**：通过 `torch_npu.flash_attention` 直调获取 `attention_out` 与 `lse_out`
- **精度对比**：按浮点误差（mean(abs diff)）比较

## 当前实现范围
- 布局：TND（变长）
- 数据类型：BF16 / FP16
- 支持 causal / 非 causal；`lse_layout="NT"`（输出 `[N,T]`）

## 文件结构
- test_run.sh                       # 执行脚本
- flash_attention_golden.py     # cpu侧 golden
- flash_attention_process.py    # 调用算子获取 npu 输出
- flash_attention_paramset.py   # 用例入参配置
- result_compare_method.py          # 精度对比
- test_flash_attention_single.py# pytest测试单用例运行主程序
- utils.py                          # 结果保存工具
- pytest.ini                        # 创建测试标记

## 使用方法
在 pytest 文件夹路径下执行：
```bash
bash test_run.sh single
```
支持自定义结果保存路径：
```bash
bash test_run.sh single -R "./result/fa_result.xlsx"
```
