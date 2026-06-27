/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_tiling.h
 * \brief Host-side TilingData struct for AddRmsNormBiasDynamicQuantAG
 *
 * 布局与旧融合算子一致: MC2 字段必须在前 (kernel base 类用 offsetof 定位 mc2CcTiling)。
 * 新增 rowBatchSize: 每批聚合写 GM 的行数 (UB 剩余空间动态计算)。
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_TILING_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_TILING_H_

#include "tiling/tiling_api.h"

struct AddRmsNormBiasDynamicQuantAGTilingData {
    // ===== MC2 / AG 通信字段 (必须在前, 供 Hccl::SetCcTilingV2 offsetof 使用) =====
    AscendC::tiling::Mc2InitTiling mc2InitTiling;
    AscendC::tiling::Mc2CcTiling mc2CcTiling;
    uint64_t groupSize;
    uint64_t rowLen;          // x1 最后一维 (= numCol, gamma 视为 1-D)
    uint64_t rowTotalNum;     // x1 除最后一维外的乘积 (AG 字节布局用)

    // ===== 计算字段 =====
    uint32_t numRow;          // 行数 (x1 前 x1DimNum-gammaDimNum 维乘积)
    uint32_t numCol;          // 列数 (= gamma 元素数)
    float    epsilon;         // rmsnorm 数值稳定项
    float    avgFactor;       // 1.0 / numCol
    uint32_t dstType;         // 量化输出类型 (固定 DT_INT8)
    uint32_t coreNum;         // 实际使用核数
    uint32_t headCoreNum;     // 头核数 (SingleN 分配)
    uint32_t rowPerHeadCore;  // 头核行数 (SingleN)
    uint32_t rowPerTailCore;  // 尾核行数 (SingleN)
    uint32_t multiRowNum;     // 兼容字段 (恒 1)
    uint32_t ubFactor;        // 对齐后的列数 AlignUp<16>(numCol)
    // MODE_NORMAL 字段
    uint32_t blockFactor;     // 每核行数 (头核)
    uint32_t latsBlockFactor; // 末核行数
    uint32_t rowFactor;       // 兼容字段
    uint32_t rowLoop;
    uint32_t rowTail;
    uint32_t lastBlockRowLoop;
    uint32_t lastBlockRowTail;
    uint32_t numColAlign;     // AlignUp<16>(numCol)
    uint32_t hasX2;           // 1 = 提供 x2
    uint32_t hasBias;         // 1 = 提供 bias
    uint32_t rowBatchSize;    // 【新】每批聚合写 GM 的行数
};

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_TILING_H_
