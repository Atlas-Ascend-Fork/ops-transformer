/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QBSA_BLOCK_TRAITS_H
#define QBSA_ARCH92_QBSA_BLOCK_TRAITS_H

#define CHILD_SPEC_TEMPLATE_ARGS \
    INPUT_T, T, layout, kvLayout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasAtten, OUTPUT_T, \
        hasLse, isPa, useDn

#define TEMPLATE_INTF \
    template <typename INPUT_T, typename T, QBSALayout layout, QBSALayout kvLayout, S1TemplateType s1TemplateType, \
              S2TemplateType s2TemplateType, DTemplateType dTemplateType, DTemplateType dVTemplateType, bool hasAtten, \
              typename OUTPUT_T, bool hasLse, bool isPa, bool useDn = true>

#define TEMPLATE_INTF_ARGS \
    INPUT_T, T, layout, kvLayout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasAtten, OUTPUT_T, \
        hasLse, isPa, useDn

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_T) \
    X(T) \
    X(OUTPUT_T)

#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

/* 2. Generate Template without default values. */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,

#define TEMPLATES_DEF_NO_DEFAULT \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. Generate Args with default values and without ChildClass. */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/* 4. Generate the BASE Template with default values and ChildClass. */

#define TEMPLATES_DEF_BASE \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 5. Generate the BASE Template without default values and with ChildClass. */

#define TEMPLATES_DEF_BASE_NO_DEFAULT \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 6. Generate BaseArgs for BASE with ChildClass. */
#define TEMPLATE_BASE_ARGS \
    ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif // QBSA_ARCH92_QBSA_BLOCK_TRAITS_H
