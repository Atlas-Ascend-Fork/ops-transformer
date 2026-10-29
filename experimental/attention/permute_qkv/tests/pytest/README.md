# permute_qkv算子测试框架

## 功能说明
基于pytest测试框架，实现permute_qkv算子的功能验证：
- **CPU侧**：复刻源 test 的 `torch_permute` 参考实现生成 golden
- **NPU侧**：通过 `torch_npu.permute_qkv` 直调获取实际数据
- **精度对比**：重排为纯数据搬移，golden 与 NPU 输出展平后应完全一致

## 使用方法
在 pytest 文件夹路径下执行：
```bash
bash test_run.sh single
```
