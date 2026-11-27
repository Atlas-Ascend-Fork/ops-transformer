# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_ep_combine_def.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE moe_ep_combine moe_ep_combine ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(moe_ep_combine_depends mc2/common mc2/3rd PARENT_SCOPE)

    foreach(CONDITION_UNIT IN LISTS ASCEND_COMPUTE_UNIT)
        if("${CONDITION_UNIT}" STREQUAL "ascend950")
            add_ops_compile_options(
                OP_NAME MoeEpCombine
                COMPUTE_UNIT Ascend950PR_9599
                OPTIONS --cce-auto-sync=off
                        -mllvm -cce-aicore-dcci-before-kernel-end=false
            )
        elseif("${CONDITION_UNIT}" STREQUAL "ascend960dt")
            add_ops_compile_options(
                OP_NAME MoeEpCombine
                COMPUTE_UNIT Ascend960DT_968B7
                OPTIONS --cce-auto-sync=off
                        -mllvm -cce-aicore-dcci-before-kernel-end=false
            )
        elseif("${CONDITION_UNIT}" STREQUAL "ascend960pr")
            add_ops_compile_options(
                OP_NAME MoeEpCombine
                COMPUTE_UNIT Ascend960PR_969A1
                OPTIONS --cce-auto-sync=off
                        -mllvm -cce-aicore-dcci-before-kernel-end=false
            )
        endif()
    endforeach()
else()
    add_mc2_modules_sources(OPTYPE moe_ep_combine moe_ep_combine ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
