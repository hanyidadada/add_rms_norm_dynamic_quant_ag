/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file reduce_common.h
 * \brief Shared reduce helpers (与旧融合算子一致, WholeReduceSum based)。
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_REDUCE_COMMON_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_REDUCE_COMMON_H_

#include "kernel_operator.h"

using namespace AscendC;

// FP32 树形求和 (WholeReduceSum), 输出单个标量到 dst_local[0]。
__aicore__ inline void ReduceSumFP32(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local, int32_t count)
{
    constexpr int32_t NUM_PER_REP_FP32 = 64;   // ONE_REPEAT_BYTE_SIZE / sizeof(float)
    constexpr int32_t NUM_PER_BLK_FP32 = 8;
    constexpr float ZERO_F = 0.0f;

    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO_F, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (likely(repeatTimes > 0)) {
        Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst_local, work_local, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#elif !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    WholeReduceSum<float, false>(dst_local, work_local, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumCustom(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local, int32_t count)
{
    ReduceSumFP32(dst_local, src_local, work_local, count);
}

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_REDUCE_COMMON_H_
