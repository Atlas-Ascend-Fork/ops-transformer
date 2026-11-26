# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU Golden 后端 — 在 NPU 上跑向量化参考实现（按行分块、头维批量）。

与大 case 适配：行块大小按 ``chunk_bytes`` 控制，峰值显存与 S1*S2 解耦；
与 CPU golden 共用同一份实现（`golden_truth.compute`），只是 device=npu，
因此同样的输入与参数得到同一口径的结果（fp32 累加，fp64 在设备上不支持）。

对外仍表现为 CPU golden（返回 CPU 张量），与 `--save_golden` 的 golden.pt 格式一致。
"""

from typing import Any, Dict, Optional

import torch

from .executor import Backend
from .golden_truth import GOLDEN_CHUNK_DEVICE

_DEVICE_CHUNK = GOLDEN_CHUNK_DEVICE


class NPUGoldenBackend(Backend):
    name = "npu_golden"

    def __init__(
        self,
        device_id: int = 0,
        chunk_bytes: int = _DEVICE_CHUNK,
        p_dtype=None,
        p_block: Optional[int] = None,
        softmax_mode: str = "auto",
        softmax_block_size: int = 0,
        kv_loader: str = "dense",
    ):
        self._device = torch.device("cpu")
        self._npu = torch.device(f"npu:{device_id}")
        self._chunk_bytes = chunk_bytes
        self._p_dtype = p_dtype
        self._p_block = p_block or 0
        self._softmax_mode = softmax_mode
        self._softmax_block_size = softmax_block_size
        self._kv_loader = kv_loader

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return torch.npu.is_available()

    def compute(
        self,
        inputs: Dict[str, torch.Tensor],
        params: Dict[str, Any],
        keep_on_device: bool = False,
    ) -> Dict[str, torch.Tensor]:
        from harness.ops.flash_attn import golden_truth

        try:
            result = golden_truth.compute(
                inputs,
                params,
                compute_dtype="fp32",
                chunk_bytes=self._chunk_bytes,
                device="npu",
                p_dtype=self._p_dtype,
                p_block=self._p_block,
                softmax_mode=self._softmax_mode,
                softmax_block_size=self._softmax_block_size,
                kv_loader=self._kv_loader,
                keep_on_device=True,
            )
            if keep_on_device:
                return {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in result.items()
                }
            return {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in result.items()
            }
        finally:
            self.clear_cache()

    def clear_cache(self):
        if torch.npu.is_available():
            torch.npu.synchronize()
            torch.npu.empty_cache()
