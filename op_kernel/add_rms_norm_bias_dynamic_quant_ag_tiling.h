/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_tiling.h
 * \brief Kernel-side TilingData struct for AddRmsNormBiasDynamicQuantAG
 *        布局与 host 侧 add_rms_norm_bias_dynamic_quant_ag_tiling.h 完全一致。
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_KERNEL_TILING_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_KERNEL_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

typedef struct {
    // ===== MC2 / AG 通信字段 (必须在前) =====
    AscendC::tiling::Mc2InitTiling mc2InitTiling;
    AscendC::tiling::Mc2CcTiling mc2CcTiling;
    uint64_t groupSize;
    uint64_t rowLen;
    uint64_t rowTotalNum;

    // ===== 计算字段 =====
    uint32_t numRow;
    uint32_t numCol;
    float    epsilon;
    float    avgFactor;
    uint32_t dstType;
    uint32_t coreNum;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNum;
    uint32_t ubFactor;
    uint32_t blockFactor;
    uint32_t latsBlockFactor;
    uint32_t rowFactor;
    uint32_t rowLoop;
    uint32_t rowTail;
    uint32_t lastBlockRowLoop;
    uint32_t lastBlockRowTail;
    uint32_t numColAlign;
    uint32_t hasX2;
    uint32_t hasBias;
    uint32_t rowBatchSize;
} AddRmsNormBiasDynamicQuantAGTilingData;

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_KERNEL_TILING_H_
