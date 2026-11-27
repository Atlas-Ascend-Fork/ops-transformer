#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import hashlib
import os
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_grouped_matmul_add_v2")
class MethodTorchGroupedMatmulAddApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupedMatmulAddApi, self).__init__(task_result)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList, aclTensor *yRef, bool transposeX, bool transposeWeight, int64_t groupType, int64_t groupListType, uint64_t *workspaceSize, aclOpExecutor **executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        group_list = input_data.kwargs["groupList"]
        y = input_data.kwargs["y"]
        list_type = input_data.kwargs["groupListType"]
        transpose_x = input_data.kwargs["transposeX"]
        print(
            f"REFERENCE_DEVICE={self.device} X_DTYPE={x.dtype} Y_DTYPE={y.dtype}",
            flush=True,
        )
        if self.device == "npu":
            torch.npu.matmul.allow_hf32 = False
            print(
                f"NPU_MATMUL_ALLOW_HF32={torch.npu.matmul.allow_hf32} MATMUL_INPUT_DTYPE=float32 ACCUM_DTYPE=float32",
                flush=True,
            )
        high_precision = x.dtype == torch.float32
        if high_precision:
            x, weight, y = x.cpu(), weight.cpu(), y.cpu()
        compute_dtype = torch.float64 if high_precision else torch.float32
        values = group_list.tolist()
        expected = [32, 64] if list_type == 0 else [32, 32]
        if values != expected:
            raise RuntimeError(f"unexpected reference groups: {values}")
        print(
            f"REFERENCE_GROUPS={values} type={list_type} compute={compute_dtype} y_input={y.dtype}",
            flush=True,
        )
        results = []
        start = 0
        for value in values:
            end = int(value) if list_type == 0 else start + int(value)
            a = (x[start:end, :].t() if transpose_x else x[:, start:end]).to(
                compute_dtype
            )
            w = weight[start:end, :].to(compute_dtype)
            results.append(torch.matmul(a, w))
            start = end
        result = torch.stack(results).reshape(y.shape) + y.to(compute_dtype)
        return result if high_precision else result.to(y.dtype)

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        # Generate non-grid inputs once in the declared dtype, then promote those same values.
        for index, name in enumerate(("x", "weight", "y")):
            original = input_data.kwargs[name]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(20261120 + index)
            dtype_name = os.environ["GMM_ATK_INPUT_DTYPE"]
            quant_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
            raw = (
                torch.rand(
                    tuple(original.shape), generator=generator, dtype=torch.float32
                )
                * 2
                - 1
            )
            value = (raw if name == "y" else raw.to(quant_dtype)).to(original.dtype)
            digest = hashlib.sha256(
                value.float().contiguous().numpy().tobytes()
            ).hexdigest()
            print(
                f"FIXED_INPUT {name} shape={tuple(value.shape)} dtype={value.dtype} sha256={digest}",
                flush=True,
            )
            input_data.kwargs[name] = value.to(original.device)
        lengths = [32, 32]
        if input_data.kwargs["groupListType"] == 0:
            values = [32, 64]
        else:
            values = lengths
        group_list = torch.tensor(values, dtype=torch.int64)
        input_data.kwargs["groupList"] = (
            group_list.npu() if self.device == "pyaclnn" else group_list.cpu()
        )


@register("aclnn_grouped_matmul_add_v2")
class AclnnFunctionApi(AclnnBaseApi):
    # def __call__(self):
    #     super().__call__()

    def init_by_input_data(self, input_data):
        for name in ("x", "weight", "y", "groupList"):
            value = input_data.kwargs[name].detach().cpu().contiguous()
            canonical = value if name == "groupList" else value.float()
            digest = hashlib.sha256(canonical.numpy().tobytes()).hexdigest()
            print(
                f"CANDIDATE_INPUT {name} shape={tuple(value.shape)} sha256={digest}",
                flush=True,
            )
        group_values = input_data.kwargs["groupList"].tolist()
        list_type = input_data.kwargs["groupListType"]
        expected = [32, 64] if list_type == 0 else [32, 32]
        if group_values != expected:
            raise RuntimeError(f"unexpected candidate groups: {group_values}")
        print(f"CANDIDATE_GROUPS={group_values} type={list_type}", flush=True)
        input_args, output_packages = super().init_by_input_data(input_data)
        # 把最后一个入参去除（默认处理会把标杆的输出拼到入参后，但实际上算子并无此参数）
        for i in range(len(output_packages)):
            input_args.pop()
        # 将前1个输入参数标记为输出
        output_packages[:] = [input_args[3]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        # logging.info(f"after_call output :{output}")
        return output
