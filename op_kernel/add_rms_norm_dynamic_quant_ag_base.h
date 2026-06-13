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
 * \file add_rms_norm_dynamic_quant_ag_base.h
 * \brief Base utilities and AG communication for AddRmsNormDynamicQuantAG fusion kernel
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_AG_BASE_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_AG_BASE_H_

#include "kernel_operator.h"
#include "add_rms_norm_dynamic_quant_ag_tiling.h"

using namespace AscendC;

#if __CCE_AICORE__ != 220
#define bfloat16_t int16_t
#endif

// ========== Constants ==========

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NUM_PER_REP_FP32 = 64;   // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr int32_t NUM_PER_REP_HALF = 128;  // ONE_REPEAT_BYTE_SIZE / sizeof(half)
constexpr int32_t BLOCK_ALIGN_NUM = 16;
constexpr float ZERO_F = 0.0f;
constexpr float ONE_F = 1.0f;
constexpr float MINUS_HALF_F = -0.5f;

// DynamicQuant constants
constexpr float DYNAMIC_QUANT_INT8_SYM_SCALE = 127.0f;
constexpr float DYNAMIC_QUANT_INT8_RECIP_SCALE = 1.0f / 127.0f;
constexpr float DYNAMIC_QUANT_EPSILON = 1e-12f;

// AG constants
constexpr static int32_t FLAG_OFFSET = 100 * 1024 * 1024;
constexpr static int32_t USED_UB_SIZE = 160 * 1024;

// ========== Type Traits ==========

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

// ========== Utility Functions ==========

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

// ========== Data Copy ==========

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t count)
{
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyParams customCopyParams;
    customCopyParams.blockLen = count * sizeof(T);
    customCopyParams.blockCount = 1;
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
        DataCopyPadParams customPadParams;
        DataCopyPad(dstTensor, srcTensor, customCopyParams, customPadParams);
    } else {
        DataCopyPad(dstTensor, srcTensor, customCopyParams);
    }
#else
    int32_t customNumPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (count % customNumPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
            int32_t customNum = AlignUp<customNumPerBlock>(count);
            DataCopy(dstTensor, srcTensor, customNum);
        } else {
            int32_t customNum = count / customNumPerBlock * customNumPerBlock;
            DataCopy(dstTensor, srcTensor, customNum);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (int32_t i = 0; i < customNumPerBlock; i++) {
                T tensorValue = srcTensor.GetValue(count - customNumPerBlock + i);
                srcTensor.SetValue(i, tensorValue);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(dstTensor[count - customNumPerBlock], srcTensor, customNumPerBlock);
        }
    }
#endif
}

// ========== Reduce Operations ==========

__aicore__ inline void ReduceSumCustom(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local, int32_t count)
{
    uint64_t reduceMask = NUM_PER_REP_FP32;
    int32_t reduceRepeatTimes = count / NUM_PER_REP_FP32;
    int32_t reduceTailCount = count % NUM_PER_REP_FP32;
    int32_t reduceBodyCount = reduceRepeatTimes * NUM_PER_REP_FP32;

    BinaryRepeatParams reduceRepeatParams;
    reduceRepeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    reduceRepeatParams.src0BlkStride = 1;
    reduceRepeatParams.src1RepStride = 0;
    reduceRepeatParams.src1BlkStride = 1;
    reduceRepeatParams.dstRepStride = 0;
    reduceRepeatParams.dstBlkStride = 1;

    Duplicate(work_local, ZERO_F, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (likely(reduceRepeatTimes > 0)) {
        Add(work_local, src_local, work_local, reduceMask, reduceRepeatTimes, reduceRepeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(reduceTailCount != 0)) {
        Add(work_local, src_local[reduceBodyCount], work_local, reduceTailCount, 1, reduceRepeatParams);
        PipeBarrier<PIPE_V>();
    }
    BlockReduceSum(dst_local, work_local, 1, reduceMask, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BlockReduceSumFP32(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local, int32_t count)
{
    int32_t blockRepeatTimes = count / NUM_PER_REP_FP32;
    int32_t blockTailCount = count % NUM_PER_REP_FP32;
    int32_t dstAddr = blockRepeatTimes * NUM_PER_BLK_FP32;
    int32_t srcAddr = blockRepeatTimes * NUM_PER_REP_FP32;
    if (likely(blockRepeatTimes > 0)) {
        BlockReduceSum(dst_local, src_local, blockRepeatTimes, NUM_PER_REP_FP32, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
    }
    if (blockTailCount != 0) {
        BlockReduceSum(dst_local[dstAddr], src_local[srcAddr], 1, blockTailCount, 1, 1, DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& src_local, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;       // count / 64
    uint64_t offsetsFp32 = repsFp32 << 6; // repsFp32 * 64
    uint64_t remsFp32 = count & 0x3f;     // count % 64

    if (likely(repsFp32 > 1)) {
        Max(src_local, src_local[NUM_PER_REP_FP32], src_local, NUM_PER_REP_FP32, repsFp32 - 1,
            {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(src_local, src_local[offsetsFp32], src_local, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = repsFp32 > 0 ? NUM_PER_REP_FP32 : count;
    WholeReduceMax(src_local, src_local, mask, 1, 8, 1, 8);
    PipeBarrier<PIPE_V>();
}

// ========== Quantization ==========

__aicore__ inline void QuantizeFp32ToInt8(
    const LocalTensor<int8_t>& outInt8,
    const LocalTensor<float>& xFp32,
    const LocalTensor<int32_t>& tmpInt32,
    const LocalTensor<half>& tmpHalf,
    uint32_t count)
{
    Cast(tmpInt32, xFp32, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();

    SetDeqScale(static_cast<half>(1.0f));
    PipeBarrier<PIPE_V>();

    Cast(tmpHalf, tmpInt32, RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();

    Cast(outInt8, tmpHalf, RoundMode::CAST_TRUNC, count);
    PipeBarrier<PIPE_V>();
}

// ========== AG: CopyGMToGM_SplitBytes (ping-pong GM-to-GM copy) ==========

__aicore__ inline void CopyGMToGM_SplitBytes(
    AscendC::GlobalTensor<int8_t> &dst1,
    AscendC::GlobalTensor<int8_t> &dst2,
    AscendC::GlobalTensor<int8_t> &src,
    const uint32_t rowLen,
    const uint32_t rowTotalNum,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &copyBuf)
{
    constexpr uint32_t EVENT_ID0 = 0;
    constexpr uint32_t EVENT_ID1 = 1;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    int64_t part1Bytes = rowLen * rowTotalNum;
    int64_t part2Bytes = rowTotalNum * sizeof(float);
    int64_t totalBytes = part1Bytes + part2Bytes;

    uint32_t tmpBufferLen = USED_UB_SIZE / 2;
    constexpr int32_t BufferNum = 2;

    AscendC::LocalTensor<int8_t> ubBase = copyBuf.Get<int8_t>();
    AscendC::LocalTensor<int8_t> buf1 = ubBase;
    AscendC::LocalTensor<int8_t> buf2 = ubBase[tmpBufferLen];

    int pingpongId = 0;
    uint32_t ubMoveBytes = tmpBufferLen;
    auto processCount = CeilDiv<int64_t>(totalBytes, ubMoveBytes);

    for (uint32_t i = 0; i < processCount; ++i) {
        uint32_t curBytes = (i == processCount - 1)
                           ? totalBytes - i * ubMoveBytes
                           : ubMoveBytes;

        uint32_t copyBytes = static_cast<uint32_t>(curBytes);
        AscendC::TEventID eventId = (pingpongId == 0) ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<int8_t> ub = (pingpongId == 0) ? buf1 : buf2;

        uint32_t srcOffset = i * ubMoveBytes;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        AscendC::DataCopyExtParams readParams(1, copyBytes, 0, 0, 0);
        AscendC::DataCopyPadExtParams<int8_t> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(ub, src[srcOffset], readParams, padParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        if (srcOffset < part1Bytes) {
            uint32_t part1End = part1Bytes;
            uint32_t blockEnd = srcOffset + curBytes;
            if (blockEnd <= part1End) {
                AscendC::DataCopyExtParams writeParams(1, copyBytes, 0, 0, 0);
                AscendC::DataCopyPad(dst1[srcOffset], ub, writeParams);
            } else {
                uint32_t bytesInPart1 = part1End - srcOffset;
                uint32_t bytesInPart2 = curBytes - bytesInPart1;
                AscendC::DataCopyExtParams p1Params(1, bytesInPart1, 0, 0, 0);
                AscendC::DataCopyPad(dst1[srcOffset], ub, p1Params);
                AscendC::DataCopyExtParams p2Params(1, bytesInPart2, 0, 0, 0);
                AscendC::DataCopyPad(dst2[0], ub[bytesInPart1], p2Params);
            }
        } else {
            uint32_t dst2Offset = srcOffset - part1Bytes;
            AscendC::DataCopyExtParams writeParams(1, copyBytes, 0, 0, 0);
            AscendC::DataCopyPad(dst2[dst2Offset], ub, writeParams);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// ========== AG Base Class ==========

class KernelAddRmsNormDynamicQuantAGBase {
protected:
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GM_ADDR buff[16];
    GM_ADDR y1Out;
    GM_ADDR scale1Out;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuf, flagBuf;
    uint32_t rankId;
    uint32_t groupSize;
    uint32_t rowLen;
    uint32_t rowTotalNum;
    uint32_t rowWork_;  // rows assigned to this core (0 for extra AG-only cores)
    int32_t blockIdx_;

public:
    __aicore__ inline KernelAddRmsNormDynamicQuantAGBase() {}

    __aicore__ inline void InitAGParams(const AddRmsNormDynamicQuantAGTilingData* tiling)
    {
        this->groupSize = tiling->groupSize;
        this->rowLen = tiling->rowLen;
        this->rowTotalNum = tiling->rowTotalNum;

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        this->hccl_.InitV2(contextGM0, tiling);
        this->hccl_.SetCcTilingV2(offsetof(AddRmsNormDynamicQuantAGTilingData, mc2CcTiling));
        for (int i = 0; i < tiling->groupSize; i++) {
            this->buff[i] = (GM_ADDR)this->hccl_.GetWindowsInAddr(i);
        }
        this->rankId = this->hccl_.GetRankId();
    }

    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buffPtr, int32_t flag)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = flagBuf.Get<int32_t>();
        ubTensor(0) = flag;
        DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0);
        GlobalTensor<int32_t> gmTensor;
        gmTensor.SetGlobalBuffer(buffPtr);
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }

    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buffPtr, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = flagBuf.Get<int32_t>();
        while (true) {
            DataCopyExtParams dataCopyParams(1, sizeof(int32_t), 0, 0, 0);
            DataCopyPadExtParams<int32_t> padParams;
            GlobalTensor<int32_t> gmTensor;
            gmTensor.SetGlobalBuffer(buffPtr);
            DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
            if (ubTensor(0) == flag) {
                break;
            }
        }
    }

    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (blockIdx_ == 0) {
            SetBuffFlag((__gm__ int32_t *)(buff[this->rankId] + FLAG_OFFSET + flag_idx * sizeof(int32_t)), flag_data);
        }
        if (blockIdx_ < this->groupSize) {
            CheckBuffFlag((__gm__ int32_t *)(buff[blockIdx_] + FLAG_OFFSET + flag_idx * sizeof(int32_t)), flag_data);
        }
    }

    __aicore__ inline void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (blockIdx_ == 0) {
                SetBuffFlag((__gm__ int32_t *)(buff[this->rankId] + FLAG_OFFSET + idx * sizeof(int32_t)), 0);
            }
        }
    }

    __aicore__ inline void ProcessAG()
    {
        this->ResetIpcFlags(2);
        if (this->blockIdx_ < this->groupSize) {
            AscendC::SyncAll<true>();
            this->CrossRankSyncV1(0, 1);
            AscendC::SyncAll<true>();

            // Only copy data if this core actually computed rows
            uint64_t tensorLen = this->rowLen * this->rowTotalNum * sizeof(int8_t);
            uint64_t scaleLen = this->rowTotalNum * sizeof(float);
            AscendC::GlobalTensor<int8_t> srcTensor;
            AscendC::GlobalTensor<int8_t> dstTensor;
            AscendC::GlobalTensor<int8_t> dstScale;
            srcTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(this->buff[this->blockIdx_]));
            dstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(this->y1Out + this->blockIdx_ * tensorLen));
            dstScale.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(this->scale1Out + this->blockIdx_ * scaleLen));

            CopyGMToGM_SplitBytes(dstTensor, dstScale, srcTensor,
                                    this->rowLen, this->rowTotalNum, this->copyBuf);
            AscendC::SyncAll<true>();
            this->CrossRankSyncV1(1, 2);
            AscendC::SyncAll<true>();
        } else {
            AscendC::SyncAll<true>();
            AscendC::SyncAll<true>();
            AscendC::SyncAll<true>();
            AscendC::SyncAll<true>();
        }
        PipeBarrier<PIPE_ALL>();
        this->hccl_.Finalize();
    }
};

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_AG_BASE_H_
