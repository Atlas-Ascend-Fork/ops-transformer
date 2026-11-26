# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from harness.device_spec import resolve_peak_gbs, spec_for_model


def test_device_spec_lookup_known_model():
    spec = spec_for_model("Ascend950PR_9579")
    assert spec is not None
    assert spec.peak_gbs == 1600.0
    assert spec.standard_freq_mhz == 1650.0


def test_device_spec_unknown_model():
    assert spec_for_model("unknown") is None
    assert spec_for_model(None) is None


def test_peak_gbs_keeps_explicit_value():
    assert resolve_peak_gbs(0, 42.0) == 42.0
