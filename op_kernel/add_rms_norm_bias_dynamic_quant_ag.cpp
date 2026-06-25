/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag.cpp
 * \brief Main kernel entry for AddRmsNormBiasDynamicQuantAG fusion operator.
 *
 * Tiling Key Encoding: (dtype_key * 10 + mode_key)
 *   dtype_key: 1 = half, 3 = bfloat16
 *   mode_key:  0 = Normal, 3 = SingleN
 *
 * | Key | Template | Data Type |
 * |-----|----------|-----------|
 * | 10  | Normal   | half      |
 * | 30  | Normal   | bfloat16  |
 * | 13  | SingleN  | half      |
 * | 33  | SingleN  | bfloat16  |
 */

#include "add_rms_norm_bias_dynamic_quant_ag_normal.h"
#include "add_rms_norm_bias_dynamic_quant_ag_single_n.h"

using namespace AscendC;

// Macro for dispatching kernel instantiation
#define FUSION_AG_OP_IMPL(templateClass, T)          \
    do {                                               \
        templateClass<T> op(&pipe);                    \
        op.Init(x1, x2, gamma, bias,                  \
                yQuant, scale, x,                     \
                workspace, &tilingData);               \
        op.Process();                                  \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_bias_dynamic_quant_ag(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR bias,
    GM_ADDR yQuant, GM_ADDR scale, GM_ADDR x,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(AddRmsNormBiasDynamicQuantAGTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddRmsNormBiasDynamicQuantAGTilingData, tilingData, tiling);

    // Dispatch by tiling key
    if (TILING_KEY_IS(10)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGNormal, half);
    } else if (TILING_KEY_IS(30)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGNormal, bfloat16_t);
    } else if (TILING_KEY_IS(13)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGSingleN, half);
    } else if (TILING_KEY_IS(33)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGSingleN, bfloat16_t);
    }
}
