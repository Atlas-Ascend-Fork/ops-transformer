# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FP32 Golden entry points for QuantSparseFlashMla."""

import logging

from qsmla_ttk_golden import (
    aclnn_arguments,
    cpu_aclnn_quant_sparse_flash_mla,
    reference_from_inputs,
)

logger = logging.getLogger(__name__)


def reference(q, quant_mode, *, testcase_name=None, **kwargs):
    logger.info("[%s] QSMLA CPU FP32 golden from final API inputs", testcase_name)
    return reference_from_inputs(q, quant_mode, promote=True, **kwargs)


def aclnn_reference(*args, **kwargs):
    values = aclnn_arguments(cpu_aclnn_quant_sparse_flash_mla, args, kwargs)
    values.pop("attn_out", None)
    return reference(**values)
