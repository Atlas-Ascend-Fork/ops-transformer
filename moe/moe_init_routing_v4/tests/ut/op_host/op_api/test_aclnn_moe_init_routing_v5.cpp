/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_moe_init_routing_v5.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

TEST(AclnnMoeInitRoutingV5, scale_alg_validation)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto callV5 = [&](int64_t quantMode, int64_t scaleAlg) {
        return aclnnMoeInitRoutingV5GetWorkspaceSize(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, -1, -1, 0, 0,
                                                     false, quantMode, nullptr, 0, scaleAlg, nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, &workspaceSize, &executor);
    };
    EXPECT_EQ(callV5(10, -1), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(callV5(10, 2), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(callV5(-1, 1), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(callV5(10, 1), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(callV5(10, 0), ACLNN_ERR_PARAM_NULLPTR);
}

namespace {
struct Hif4UtTensorParam {
    aclDataType xDtype;
    aclDataType expandedXDtype;
    int64_t quantMode;
};

// HIF4 门禁用例：非 Ascend960 平台、或 quantMode 与 expandedX dtype 不匹配时，均应在 aclnn 层报错
aclnnStatus RunHif4GateCase(const Hif4UtTensorParam& param)
{
    auto x = TensorDesc({2, 128}, param.xDtype, ACL_FORMAT_ND).ValueRange(-4, 4);
    auto expertIdx = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto expandedXOut = TensorDesc({4, 128}, param.expandedXDtype, ACL_FORMAT_ND);
    auto expandedRowIdxOut = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto expertTokensCountOrCumsumOut = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
    auto expandedScaleOut = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    std::vector<int64_t> activeExpertRangeArray = {0, 8};
    aclIntArray* activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());

    int64_t expertCapacity = -1;
    int64_t expertNum = 8;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    int64_t expertTokensNumFlag = true;
    int64_t quantMode = param.quantMode;
    int64_t rowIdxType = 0;
    int64_t scaleAlg = 0;

    auto ut =
        OP_API_UT(aclnnMoeInitRoutingV5,
                  INPUT(x, expertIdx, nullptr, nullptr, nullptr, nullptr, expertCapacity, expertNum, dropPadMode,
                        expertTokensNumType, expertTokensNumFlag, quantMode, activeExpertRange, rowIdxType, scaleAlg),
                  OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut, nullptr));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
}

} // namespace

// UT 使用 tests/ut/framework_normal/op_api/stub 的平台桩件（默认 SocVersion=ASCEND910B），
// 通过 op::SetPlatformNpuArch 显式指定平台，用例与真实机器型号无关。
class Hif4GateFixture : public testing::Test {
protected:
    void TearDown() override
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }
};

// 非 Ascend960 平台：quant_mode=10 被平台门禁拦截
// 注：ACL_HIFLOAT4 枚举需要较新版本的 CANN acl 头，此处以 BF16 输出构造非法组合，
// HiFloat4 输出的正向用例由 ST 覆盖。
TEST_F(Hif4GateFixture, hif4_rejected_on_non_ascend960)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);
    EXPECT_EQ(RunHif4GateCase({ACL_BF16, ACL_BF16, 10}), ACLNN_ERR_PARAM_INVALID);
}

// Ascend960 平台：quant_mode=10 但 expanded_x 不是 HiFloat4，被 dtype 一致性校验拦截
TEST_F(Hif4GateFixture, hif4_quant_mode_requires_hifloat4_output)
{
    op::SetPlatformNpuArch(NpuArch::DAV_9201);
    EXPECT_EQ(RunHif4GateCase({ACL_BF16, ACL_BF16, 10}), ACLNN_ERR_PARAM_INVALID);
}
