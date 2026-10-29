# flash_attention_grad算子测试框架

## 功能说明
基于pytest测试框架，实现flash_attention_grad算子（反向）的功能验证：
- **CPU侧**：变长 TND 因果/非因果注意力反向 golden（torch，输出 dq/dk/dv）
- **NPU侧**：通过 `torch_npu.flash_attention_grad` 直调获取 `dq/dk/dv`
- **精度对比**：按浮点误差（mean(abs diff)）比较

## 当前实现范围
- 布局：TND（变长），`lse_layout="NT"`（softmax_lse 形状 `[N,T]`）
- 数据类型：BF16 / FP16
- 支持 causal / 非 causal

## 文件结构
- test_run.sh                           # 执行脚本
- flash_attention_grad_golden.py    # cpu侧 golden
- flash_attention_grad_process.py   # 调用算子获取 npu 输出
- flash_attention_grad_paramset.py  # 用例入参配置
- result_compare_method.py              # 精度对比
- test_flash_attention_grad_single.py # pytest测试单用例运行主程序
- utils.py                              # 结果保存工具
- pytest.ini                            # 创建测试标记

## 使用方法
在 pytest 文件夹路径下执行：
```bash
bash test_run.sh single
```
