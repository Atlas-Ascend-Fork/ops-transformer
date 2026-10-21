/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_FP8_ARCH92_TILING_H_
#define QBSA_FP8_ARCH92_TILING_H_

#include "../../arch35/qbsa_fp8/qbsa_fp8_tiling.h"
#include "../../../op_kernel/arch92/qbsa_fp8/qbsa_fp8_tiling_data.h"

namespace optiling {
// FP8 input contracts are shared; arch92 has independent task geometry and payload capacity.
class QbsaFp8Arch92Tiling final : public QbsaFp8Arch35Tiling {
public:
    explicit QbsaFp8Arch92Tiling(gert::TilingContext* context)
        : QbsaFp8Arch35Tiling(context)
    {}

private:
    ge::graphStatus BuildTilingData() override;
    ge::graphStatus SaveTilingData() override;
    QuantBlockSparseAttnFp8Arch92TilingData arch92TilingData_{};
};
} // namespace optiling
#endif // QBSA_FP8_ARCH92_TILING_H_
