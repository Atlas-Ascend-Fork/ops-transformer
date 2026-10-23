/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class SplitQkvRmsnormMrope : public OpDef {
public:
    explicit SplitQkvRmsnormMrope(const char *name) : OpDef(name)
    {
        Input("qkv").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("q_weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("k_weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("cos_sin").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("q_bias").ParamType(OPTIONAL).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("k_bias").ParamType(OPTIONAL).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Output("q_output").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Output("k_output").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Output("v_output").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Output("gate_output").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Attr("num_q_heads").Int(16);
        Attr("num_kv_heads").Int(4);
        Attr("eps").Float(1.0e-6F);
        Attr("interleaved").Bool(true);
        Attr("has_gate").Bool(true);
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(SplitQkvRmsnormMrope);
} // namespace ops
