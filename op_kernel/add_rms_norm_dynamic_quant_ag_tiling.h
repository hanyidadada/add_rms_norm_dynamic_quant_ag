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
 * \file add_rms_norm_dynamic_quant_ag_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_AG_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_AG_TILING_H

typedef struct {    
    AscendC::tiling::Mc2InitTiling mc2InitTiling;
    AscendC::tiling::Mc2CcTiling mc2CcTiling;
    uint64_t useCore;             // 使用的 Core 数量
    uint64_t groupSize;             // group通信域大小
    uint64_t rowLen;
    uint64_t rowTotalNum;
    uint64_t numFirstDim;         // 第一维度的数量
    uint64_t numLastDimAligned;   // 对齐后的最后一维度数量
    uint64_t numLastDim;          // 最后一维度的数量
    uint64_t firstDimPerCoreTail; // 每个 Core 处理的第一维度尾部数量
    uint64_t firstDimPerCore;     // 每个 Core 处理的第一维度数量
    uint64_t firstDimPerLoop;     // 每次循环处理的第一维度数量
    uint64_t lastDimLoopNum;      // 最后一维度的循环次数
    uint64_t lastDimSliceLen;     // 最后一维度的切片长度
    uint64_t lastDimSliceLenTail; // 最后一维度的切片尾部长度
    uint32_t smoothNum1;          // 平滑参数 1
    uint32_t smoothNum2;          // 平滑参数 2
    float    epsilon;             // 防止除零的极小值
    int32_t  outQuant1Flag;       // 输出量化标志位 1
    int32_t  outQuant2Flag;       // 输出量化标志位 2
    float    avgFactor;           // 平均因子
    uint32_t betaFlag;            // Beta 标志位
} AddRmsNormDynamicQuantAGTilingData;

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_H
