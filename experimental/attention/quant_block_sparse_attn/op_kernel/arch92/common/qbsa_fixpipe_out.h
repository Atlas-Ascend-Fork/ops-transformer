/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file FixpipeOut.h
 * \brief
 */
#ifndef FIXPIPEOUT_H
#define FIXPIPEOUT_H

constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_UB = {
    CO2Layout::ROW_MAJOR,
    true}; // ROW_MAJOR: enable NZ2ND and output ND data; true specifies that the destination address is in UB.
constexpr FixpipeConfig FA_CFG_NZ_UB = {
    CO2Layout::NZ, true}; // Disable NZ2ND and output NZ data; true specifies that the destination address is in UB.

#endif
