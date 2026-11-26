# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in CPU three-party TestSpec for SparseFlashMla."""

import importlib.util
from functools import partial
from pathlib import Path
from typing import ClassVar

base_spec = importlib.util.spec_from_file_location(
    "smla_two_party_spec", Path(__file__).with_name("spec.py")
)
base = importlib.util.module_from_spec(base_spec)
base_spec.loader.exec_module(base)
fp32_reference = base.load_impl_module("fp32_reference")
cross_check = base.load_impl_module("cross_check")


class SparseFlashMlaThreePartySpec:
    customize_inputs = partial(
        base.inputs_module.generate_sparse_flash_mla_inputs, retain_golden=False
    )
    npu_preprocess = base.npu_preprocess_module.run
    golden = fp32_reference.reference
    third_party: ClassVar[dict[str, type]] = {"torch": cross_check.TorchReference}
    tolerance = cross_check.TOLERANCE
    pre_compare = staticmethod(cross_check.pre_compare)


class AclnnSparseFlashMlaThreePartySpec(SparseFlashMlaThreePartySpec):
    customize_inputs = partial(
        base.inputs_module.generate_aclnn_sparse_flash_mla_inputs, retain_golden=False
    )
    npu_preprocess = base.npu_preprocess_module.run_aclnn
    golden = fp32_reference.aclnn_reference
    third_party: ClassVar[dict[str, type]] = {"torch": cross_check.AclnnTorchReference}
    pre_compare = staticmethod(partial(cross_check.pre_compare, check_empty_lse=False))


__spec__ = {
    "torch.ops.cann_ops_transformer.sparse_flash_mla": "SparseFlashMlaThreePartySpec",
    "aclnnSparseFlashMla": "AclnnSparseFlashMlaThreePartySpec",
}
