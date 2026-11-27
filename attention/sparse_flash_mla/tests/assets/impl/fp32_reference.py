# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FP32 Golden entry points for SparseFlashMla."""

import logging

from smla_ttk_golden import (
    aclnn_arguments,
    cpu_aclnn_sparse_flash_mla,
    reference_from_inputs,
)

logger = logging.getLogger(__name__)


def reference(q, *, testcase_name=None, **kwargs):
    logger.info("[%s] SMLA CPU FP32 golden from final API inputs", testcase_name)
    return reference_from_inputs(q, promote=True, **kwargs)


def aclnn_reference(*args, **kwargs):
    values = aclnn_arguments(cpu_aclnn_sparse_flash_mla, args, kwargs)
    values.pop("attn_out", None)
    return reference(**values)
