/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag.cpp
 * \brief Main kernel entry for AddRmsNormBiasDynamicQuantAG fusion operator.
 *
 * Tiling Key 组合编码: TilingKey = rmsKey * 100 + quantKey
 *   rmsKey   = add_rms_norm_bias 的 dtypeKey*10+modeKey (normal=10/30, single_n=13/33)
 *   quantKey = dynamic_quant 的 db 对称量化 key (db=3 half/2 bf16; non-db=1 half/0 bf16)
 *
 * | Key  | 分支       | Template | Data Type  |
 * |------|-----------|----------|------------|
 * | 1003 | Normal+db | Normal   | half       |
 * | 3002 | Normal+db | Normal   | bfloat16_t |
 * | 1301 | SingleN   | SingleN  | half       |
 * | 3300 | SingleN   | SingleN  | bfloat16_t |
 */

#include "add_rms_norm_bias_dynamic_quant_ag_normal.h"
#include "add_rms_norm_bias_dynamic_quant_ag_single_n.h"

using namespace AscendC;

// 派发宏: 构造 -> Init -> Process
#define FUSION_AG_OP_IMPL(templateClass, T)        \
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

    // 按 TilingKey 派发
    if (TILING_KEY_IS(1003)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGNormal, half);
    } else if (TILING_KEY_IS(3002)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGNormal, bfloat16_t);
    } else if (TILING_KEY_IS(1301)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGSingleN, half);
    } else if (TILING_KEY_IS(3300)) {
        FUSION_AG_OP_IMPL(KernelAddRmsNormBiasDynamicQuantAGSingleN, bfloat16_t);
    }
}
