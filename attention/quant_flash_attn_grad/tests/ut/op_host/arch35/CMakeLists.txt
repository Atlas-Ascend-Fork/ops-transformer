# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})

    # Use the same generated data layout and key encoding as the host tiling sources.
    if(TARGET ${OP_TILING_MODULE_NAME}_cases_obj AND DEFINED quant_flash_attn_grad_pypto_gen_dir)
        file(GLOB _qfag_key_headers "${quant_flash_attn_grad_pypto_gen_dir}/*_tilingkey.h")
        file(GLOB _qfag_data_headers "${quant_flash_attn_grad_pypto_gen_dir}/*_tiling.h")
        set(_qfag_force_options "")
        foreach(_header ${_qfag_key_headers} ${_qfag_data_headers})
            list(APPEND _qfag_force_options -include ${_header})
        endforeach()
        set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/test_quant_flash_attn_grad_tiling.cpp"
            TARGET_DIRECTORY ${OP_TILING_MODULE_NAME}_cases_obj
            PROPERTIES COMPILE_OPTIONS "${_qfag_force_options}"
        )
    endif()
endif()
