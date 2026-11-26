# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# FE compatibility registrations are linked into libopgraph_transformer.so.
file(GLOB_RECURSE OP_GRAPH_COMMON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(OP_GRAPH_COMMON_SRCS)
    add_graph_plugin_modules()
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OPS_TRANSFORMER_DIR}/common/stub/inc/op_graph)
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj SYSTEM PRIVATE
        ${ASCEND_DIR}/include ${ASCEND_DIR}/include/external ${ASCEND_DIR}/include/op_common
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/include ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/external
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/op_common)
    # 编译 legacy 平台（ascend950/ascend350/ascend960dt/ascend960pr）时注入平台宏，桩注册内容才编入
    # ascend350 与 ascend950 同属 dav_3510，OPP 白名单按 arch 3510 生效，需一并覆盖
    set(OP_GRAPH_LEGACY_SOCS "ascend950" "ascend350" "ascend960dt" "ascend960pr")
    set(_inject_legacy_stub FALSE)
    foreach(_soc IN LISTS ASCEND_COMPUTE_UNIT)
        if(_soc IN_LIST OP_GRAPH_LEGACY_SOCS)
            set(_inject_legacy_stub TRUE)
            break()
        endif()
    endforeach()
    if(_inject_legacy_stub)
        set_source_files_properties(${OP_GRAPH_COMMON_SRCS} PROPERTIES
            COMPILE_DEFINITIONS OP_GRAPH_TRANSFORMER_LEGACY=1
        )
    endif()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_GRAPH_COMMON_SRCS})
endif()
