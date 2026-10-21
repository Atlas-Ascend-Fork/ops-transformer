/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_mla_metadata_aicpu.h
 * \brief AICPU metadata producer for FlashMlaMetadata.
 */

#ifndef FLASH_MLA_METADATA_AICPU_H
#define FLASH_MLA_METADATA_AICPU_H

#include <string>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "../../common/op_kernel/load_balance/section_stream_k/section_stream_k.h"

namespace aicpu {

class FlashMlaMetadataCpuKernel : public CpuKernel {
public:
    FlashMlaMetadataCpuKernel() = default;
    ~FlashMlaMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    bool Prepare(CpuKernelContext& ctx);
    bool BalanceSchedule(load_balance::SectionStreamKResult& splitRes);
    bool GenMetadata(load_balance::SectionStreamKResult& splitRes);

    bool ParamsCheck();
    bool CheckAttrs();
    bool CheckActualKvSeq();
    bool CheckActualQuerySeq(int64_t batchSize);
    bool CheckMetadataCapacity(uint32_t sectionNum);

    bool ParamsInit();
    void InitDeviceInfo();
    void InitBaseInfo();
    void InitLoadBalanceParams();

private:
    CpuKernelContext* context_ = nullptr;
    Tensor* cuSeqlensQ_ = nullptr;
    Tensor* cacheSeqlens_ = nullptr;
    Tensor* sequsedQ_ = nullptr;
    Tensor* metadata_ = nullptr;

    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDimQk_ = 0;
    int32_t headDimV_ = 0;
    int32_t maskMode_ = 0;
    std::string layoutQ_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U; // 36: default aic num
    int32_t aivCoreNum_ = 72U; // 72: default aiv num

    bool isActualSeqlenQAccum_ = false;
    bool isActualSeqlenKvAccum_ = false;
    std::vector<int64_t> actualSeqlenQ_{};
    std::vector<int64_t> actualSeqlenKv_{};

    uint32_t mBaseSize_ = 64;   // 64: default value
    uint32_t s2BaseSize_ = 128; // 128: default value
    load_balance::DeviceInfo deviceInfo;
    load_balance::BaseInfo baseInfo;
    load_balance::SectionStreamKParam param;

private:
    enum class ParamId : uint32_t {
        cuSeqlensQ = 0,
        cacheSeqlens = 1,
        sequsedQ = 2,
        metaData = 0,
    };
};
} // namespace aicpu

#endif
