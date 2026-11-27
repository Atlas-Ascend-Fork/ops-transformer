# GroupedMatmulAddV2 ATK 双标杆测试

三份配置共 64 个精度用例，覆盖 FP16/BF16 输入、FP32/同类型 16 位输出、X 转置与否，以及两种 `groupListType`。`atk_original16_dual.json` 使用两组均匀分组 `[32, 32]`；另两份配置使用非均匀分组 `[37, 0, 23, 52]`，包括空组。`atk_many_tasks_dual.json` 使用较大的 N 维，覆盖多个调度轮次。

测试使用 ATK `cv_fused_double_benchmark`：CPU 高精度参考以 FP64 计算；独立 NPU 参考逐组使用 FP32 Matmul 和 Add，并关闭 HF32。待测算子由指定的自定义 vendor 加载。

在已配置 CANN、PyTorch、torch_npu 和 ATK 的 A2/A3 环境中，先设置自定义算子及空闲设备，再从此目录运行：

```bash
export ASCEND_CUSTOM_OPP_PATH="/path/to/vendors/<vendor>"
export ATK_CUSTOM_OPP_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
export ASCEND_RT_VISIBLE_DEVICES="<physical_device_id>"

python -c 'import os; from atk.tasks.backends.lib_interface.acl_wrapper import get_opp_lib_path; p = get_opp_lib_path("aclnnGroupedMatmulAddV2GetWorkspaceSize"); print("ATK operator library:", p); assert os.path.realpath(p) == os.path.realpath(os.environ["ATK_CUSTOM_OPP_PATH"])'

atk aclnn atk_original16_dual.json --task accuracy \
  --compare-backend npu --compare-devices 0 --devices 0 \
  --concurrency 1 --timeout 180 -p executor_dual_verified.py -o ./atk_dual_basic_output
atk aclnn atk_groups_uneven_dual.json --task accuracy \
  --compare-backend npu --compare-devices 0 --devices 0 \
  --concurrency 1 --timeout 180 -p executor_dual_verified.py -o ./atk_dual_uneven_output
atk aclnn atk_many_tasks_dual.json --task accuracy \
  --compare-backend npu --compare-devices 0 --devices 0 \
  --concurrency 1 --timeout 180 -p executor_dual_verified.py -o ./atk_dual_many_output
```

运行前应确认设备空闲；结果中需核对实际加载的自定义库、CPU/NPU 两路参考和双标杆误差指标。这三份配置只用于精度验证，不包含性能对照。
