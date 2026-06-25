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
 * \file add_rms_norm_bias_dynamic_quant_ag_tiling.h
 * \brief Kernel-side tiling data structure for AddRmsNormBiasDynamicQuantAG
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_DYN_QUANT_AG_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_DYN_QUANT_AG_TILING_H

typedef struct {
    // MC2 fields (must be first for correct offsetof)
    AscendC::tiling::Mc2InitTiling mc2InitTiling;
    AscendC::tiling::Mc2CcTiling mc2CcTiling;
    uint64_t groupSize;
    uint64_t rowLen;
    uint64_t rowTotalNum;

    // Computation fields
    uint32_t numRow;         // N - number of rows
    uint32_t numCol;         // D - number of columns
    float    epsilon;        // numerical stability term
    float    avgFactor;      // 1.0 / numCol, pre-computed mean factor
    uint32_t dstType;        // output quantization type: DT_INT8=2
    uint32_t coreNum;        // total number of cores used
    uint32_t headCoreNum;    // number of head cores (with ceil rows)
    uint32_t rowPerHeadCore; // rows per head core
    uint32_t rowPerTailCore; // rows per tail core
    uint32_t multiRowNum;    // rows processed per iteration
    uint32_t ubFactor;       // UB allocation factor (aligned col size)
    // MODE_NORMAL fields
    uint32_t blockFactor;    // rows per core (head cores)
    uint32_t latsBlockFactor;// rows for last core (tail)
    uint32_t rowFactor;      // rows per inner loop iteration (rstd batch size)
    uint32_t rowLoop;        // inner loop count (head cores)
    uint32_t rowTail;        // tail rows in last inner loop iteration (head cores)
    uint32_t lastBlockRowLoop;  // inner loop count (last core)
    uint32_t lastBlockRowTail;  // tail rows in last inner loop iteration (last core)
    uint32_t numColAlign;    // aligned numCol (to BLOCK_ALIGN_NUM)
    uint32_t hasX2;          // 1 if x2 input is provided, 0 otherwise
    uint32_t hasBias;        // 1 if bias input is provided, 0 otherwise
} AddRmsNormBiasDynamicQuantAGTilingData;

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_DYN_QUANT_AG_TILING_H
