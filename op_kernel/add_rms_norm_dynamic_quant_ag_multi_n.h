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
 * \file add_rms_norm_dynamic_quant_ag_multi_n.h
 * \brief MultiN kernel template for AddRmsNormDynamicQuantAG: process multiple rows per iteration per core.
 *        Suitable for smaller D where UB can fit multiple rows.
 *        AG (AllGather) phase appended after computation.
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_AG_MULTI_N_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_AG_MULTI_N_H_

#include "add_rms_norm_dynamic_quant_ag_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormDynamicQuantAGMultiN : public KernelAddRmsNormDynamicQuantAGBase {
    static constexpr int32_t MAX_BUFFER = 195584; // (192 - 1) * 1024 bytes
    static constexpr uint32_t SZ_FLOAT = sizeof(float);    // 4
    static constexpr uint32_t SZ_HALF  = sizeof(half);     // 2
    static constexpr uint32_t SZ_INT8  = sizeof(int8_t);   // 1
    static constexpr uint32_t T_SZ = sizeof(T);            // 2 for half, 2 for bf16

public:
    __aicore__ inline KernelAddRmsNormDynamicQuantAGMultiN(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
        GM_ADDR yQuant, GM_ADDR scale, GM_ADDR yAdd, GM_ADDR rstd,
        GM_ADDR workspace, const AddRmsNormDynamicQuantAGTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->numRow = tiling->numRow;
        this->numCol = tiling->numCol;
        this->ubFactor = tiling->ubFactor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = tiling->avgFactor;
        this->dstType = tiling->dstType;
        this->headCoreNum = tiling->headCoreNum;
        this->rowPerHeadCore = tiling->rowPerHeadCore;
        this->rowPerTailCore = tiling->rowPerTailCore;
        this->multiRowNum = tiling->multiRowNum;

        this->blockIdx_ = GetBlockIdx();

        // Determine rows for this core (head/tail distribution)
        if (this->blockIdx_ < headCoreNum) {
            this->rowWork = this->rowPerHeadCore;
        } else {
            this->rowWork = this->rowPerTailCore;
        }

        // Guard: cores beyond useCoreNum are AG-only (groupSize > useCoreNum)
        // useCoreNum = headCoreNum when rowPerTailCore==0, else numCore
        {
            uint32_t useCoreNum = (this->rowPerTailCore > 0) ? tiling->coreNum : this->headCoreNum;
            if (this->blockIdx_ >= useCoreNum) {
                this->rowWork = 0;
            }
        }

        this->rowWork_ = this->rowWork;

        // Guard: extra cores launched for AG only (groupSize > useCoreNum)
        if (this->rowWork == 0) {
            this->InitAGParams(tiling);
            this->y1Out = yQuant;
            this->scale1Out = scale;
            pPipe->InitBuffer(unitBuf, MAX_BUFFER);
            return;
        }

        // Calculate loop count
        this->rowLoop = CeilDiv(this->rowWork, this->multiRowNum);
        this->rowTail = this->rowWork - (this->rowLoop - 1) * this->multiRowNum;

        // Calculate global memory offset for this core
        uint32_t rowOffset = 0;
        if (this->blockIdx_ < headCoreNum) {
            rowOffset = this->blockIdx_ * rowPerHeadCore;
        } else {
            rowOffset = headCoreNum * rowPerHeadCore + (this->blockIdx_ - headCoreNum) * rowPerTailCore;
        }

        // === AG initialization ===
        this->InitAGParams(tiling);

        // === Save final output addresses ===
        this->y1Out = yQuant;
        this->scale1Out = scale;

        // Set up global tensors
        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + rowOffset * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + rowOffset * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);

        // === Quantized output redirected to HCCL window ===
        uint32_t offsetScale = this->rowTotalNum *  this->rowLen;
        yQuantGm.SetGlobalBuffer(
            (__gm__ int8_t*)this->buff[this->rankId] + rowOffset * numCol, rowWork * numCol);
        scaleGm.SetGlobalBuffer(
            (__gm__ float*)(this->buff[this->rankId] + offsetScale) + rowOffset, rowWork);

        yAddGm.SetGlobalBuffer((__gm__ T*)yAdd + rowOffset * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + rowOffset, rowWork);

        // Compute UB layout offsets (all in float elements for ubLocal indexing)
        ComputeUBLayout();

        pPipe->InitBuffer(unitBuf, MAX_BUFFER);
    }

    __aicore__ inline void Process()
    {
        if (this->rowWork_ == 0) {
            // Extra core (groupSize > useCoreNum): skip computation, AG sync only
            PipeBarrier<PIPE_ALL>();
            pPipe->Reset();
            pPipe->InitBuffer(this->copyBuf, USED_UB_SIZE);
            pPipe->InitBuffer(this->flagBuf, 32);
            this->ProcessAG();
            return;
        }

        if constexpr (is_same<T, half>::value) {
            ProcessFp16();
        } else {
            ProcessBf16();
        }

        // === AG phase: pipe reset + AllGather ===
        PipeBarrier<PIPE_ALL>();
        pPipe->Reset();
        pPipe->InitBuffer(this->copyBuf, USED_UB_SIZE);
        pPipe->InitBuffer(this->flagBuf, 32);
        this->ProcessAG();
    }

private:
    // ========================================================================
    // UB Layout Computation (all sizes/offsets in float elements)
    // ========================================================================

    __aicore__ inline void ComputeUBLayout()
    {
        uint32_t M = multiRowNum;

        xBlockFloatSz  = M * ubFactor * T_SZ / SZ_FLOAT;
        fp32BlockSz    = M * ubFactor;
        tmpBlockSz     = fp32BlockSz;
        rstdBlockSz    = M * NUM_PER_BLK_FP32;
        outInt8FloatSz = (M * numCol + SZ_FLOAT - 1) / SZ_FLOAT;

        off_x1      = 0;
        off_x2      = off_x1 + xBlockFloatSz;
        off_xFp32   = off_x2 + xBlockFloatSz;
        off_sqx     = off_xFp32 + fp32BlockSz;
        off_tmp     = off_sqx + fp32BlockSz;
        off_rstd    = off_tmp + tmpBlockSz;
        off_outInt8 = off_rstd + rstdBlockSz;
    }

    // ========================================================================
    // FP16 MultiN Path
    // ========================================================================
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();

        LocalTensor<T> x1Block = xLocal[off_x1 * SZ_FLOAT / T_SZ];
        LocalTensor<T> x2Block = xLocal[off_x2 * SZ_FLOAT / T_SZ];
        LocalTensor<float> xFp32Block = ubLocal[off_xFp32];
        LocalTensor<float> sqxBlock   = ubLocal[off_sqx];
        LocalTensor<float> tmpBlock   = ubLocal[off_tmp];
        LocalTensor<float> rstdBlock  = ubLocal[off_rstd];
        LocalTensor<int8_t> outInt8Block = ubLocal[off_outInt8].template ReinterpretCast<int8_t>();

        // Pre-load gamma
        event_t eventVMTE2G = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2G);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2G);
        DataCopyCustom<T>(x2Block, gammaGm, numCol);
        event_t eventMTE2VG = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2VG);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2VG);

        for (uint32_t i_o = 0; i_o < rowLoop; i_o++) {
            uint32_t curRows = (i_o == rowLoop - 1) ? rowTail : multiRowNum;
            uint32_t curElems = curRows * numCol;
            uint64_t gmBias = static_cast<uint64_t>(i_o) * static_cast<uint64_t>(multiRowNum) *
                              static_cast<uint64_t>(numCol);

            // Stage 1: Add (FP16)
            DataCopyCustom<T>(x1Block, x1Gm[gmBias], curElems);
            event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V1);

            DataCopyCustom<T>(x2Block, x2Gm[gmBias], curElems);
            event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);

            Add(x1Block, x1Block, x2Block, curElems);
            PipeBarrier<PIPE_V>();

            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyCustom<T>(yAddGm[gmBias], x1Block, curElems);
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);

            // Stage 2: RmsNorm (FP32)
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            DataCopyCustom<T>(x2Block, gammaGm, numCol);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

            Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
            PipeBarrier<PIPE_V>();

            Mul(sqxBlock, xFp32Block, xFp32Block, curElems);
            PipeBarrier<PIPE_V>();
            Muls(sqxBlock, sqxBlock, avgFactor, curElems);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < curRows; r++) {
                ReduceSumCustom(rstdBlock[r * NUM_PER_BLK_FP32],
                                sqxBlock[r * numCol], tmpBlock, numCol);
            }

            Adds(rstdBlock, rstdBlock, epsilon, curRows * NUM_PER_BLK_FP32);
            PipeBarrier<PIPE_V>();
            Sqrt(rstdBlock, rstdBlock, curRows * NUM_PER_BLK_FP32);
            Duplicate(tmpBlock, ONE_F, NUM_PER_BLK_FP32);
            PipeBarrier<PIPE_V>();

            int32_t repTimes = curRows * NUM_PER_BLK_FP32 / NUM_PER_REP_FP32;
            int32_t tailCnt  = curRows * NUM_PER_BLK_FP32 % NUM_PER_REP_FP32;
            int32_t bodyCnt  = repTimes * NUM_PER_REP_FP32;
            if (likely(repTimes > 0)) {
                Div(rstdBlock, tmpBlock, rstdBlock, NUM_PER_REP_FP32, repTimes,
                    {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
            }
            if (unlikely(tailCnt != 0)) {
                Div(rstdBlock[bodyCnt], tmpBlock, rstdBlock[bodyCnt], tailCnt, 1,
                    {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
            }
            PipeBarrier<PIPE_V>();

            event_t eventVMTE3R = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3R);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3R);
            DataCopyParams rstdCopyParams;
            rstdCopyParams.blockLen = sizeof(float);
            rstdCopyParams.blockCount = curRows;
            DataCopyPad(rstdGm[i_o * multiRowNum], rstdBlock, rstdCopyParams);

            for (uint32_t r = 0; r < curRows; r++) {
                float rstdVal = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
                Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], rstdVal, numCol);
            }
            PipeBarrier<PIPE_V>();

            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);

            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
            Cast(sqxBlock, x2Block, RoundMode::CAST_NONE, numCol);   // gamma FP16 → FP32
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < curRows; r++) {
                Mul(xFp32Block[r * numCol], sqxBlock, xFp32Block[r * numCol], numCol);  // FP32 mul
            }
            PipeBarrier<PIPE_V>();
            Cast(x1Block, xFp32Block, RoundMode::CAST_NONE, curElems);  // FP32 → FP16
            PipeBarrier<PIPE_V>();

            // Stage 3: DynamicQuant (FP32)
            Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
            PipeBarrier<PIPE_V>();

            Abs(sqxBlock, xFp32Block, curElems);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < curRows; r++) {
                LocalTensor<float> rowMaxLocal = sqxBlock[r * numCol];
                Duplicate(rstdBlock[r * NUM_PER_BLK_FP32], ZERO_F, NUM_PER_BLK_FP32);
                PipeBarrier<PIPE_V>();
                uint64_t repsFp32 = numCol >> 6;
                uint64_t offsetsFp32 = repsFp32 << 6;
                uint64_t remsFp32 = numCol & 0x3f;
                if (likely(repsFp32 > 0)) {
                    Max(rstdBlock[r * NUM_PER_BLK_FP32], rowMaxLocal, rstdBlock[r * NUM_PER_BLK_FP32],
                        NUM_PER_REP_FP32, repsFp32, {1, 1, 0, 1, DEFAULT_REPEAT_STRIDE, 0});
                    PipeBarrier<PIPE_V>();
                }
                if (unlikely(remsFp32 > 0)) {
                    Max(rstdBlock[r * NUM_PER_BLK_FP32], rowMaxLocal[offsetsFp32],
                        rstdBlock[r * NUM_PER_BLK_FP32], remsFp32, 1,
                        {1, 1, 0, 1, DEFAULT_REPEAT_STRIDE, 0});
                    PipeBarrier<PIPE_V>();
                }
                uint32_t mask = repsFp32 > 0 ? NUM_PER_REP_FP32 : numCol;
                WholeReduceMax(rstdBlock[r * NUM_PER_BLK_FP32], rstdBlock[r * NUM_PER_BLK_FP32],
                    mask, 1, 8, 1, 8);
                PipeBarrier<PIPE_V>();
            }

            for (uint32_t r = 0; r < curRows; r++) {
                float rowMax = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
                float scaleVal = rowMax * DYNAMIC_QUANT_INT8_RECIP_SCALE;
                float invScale = (rowMax > DYNAMIC_QUANT_EPSILON) ?
                    (DYNAMIC_QUANT_INT8_SYM_SCALE / rowMax) : 0.0f;
                rstdBlock.SetValue(r * NUM_PER_BLK_FP32, scaleVal);
                Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], invScale, numCol);
            }
            PipeBarrier<PIPE_V>();

            event_t eventVMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3S);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3S);
            DataCopyParams scaleCopyParams;
            scaleCopyParams.blockLen = sizeof(float);
            scaleCopyParams.blockCount = curRows;
            DataCopyPad(scaleGm[i_o * multiRowNum], rstdBlock, scaleCopyParams);

            LocalTensor<int32_t> tmpInt32Block = tmpBlock.template ReinterpretCast<int32_t>();
            LocalTensor<half> tmpHalfBlock = tmpBlock.template ReinterpretCast<half>();
            QuantizeFp32ToInt8(outInt8Block, xFp32Block, tmpInt32Block, tmpHalfBlock, curElems);

            event_t eventVMTE3Q = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Q);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Q);
            DataCopyCustom<int8_t>(yQuantGm[gmBias], outInt8Block, curElems);
        }
    }

    // ========================================================================
    // BF16 MultiN Path
    // ========================================================================
    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();

        LocalTensor<T> x1Block = xLocal[off_x1 * SZ_FLOAT / T_SZ];
        LocalTensor<T> x2Block = xLocal[off_x2 * SZ_FLOAT / T_SZ];
        LocalTensor<float> xFp32Block = ubLocal[off_xFp32];
        LocalTensor<float> sqxBlock   = ubLocal[off_sqx];
        LocalTensor<float> tmpBlock   = ubLocal[off_tmp];
        LocalTensor<float> rstdBlock  = ubLocal[off_rstd];
        LocalTensor<int8_t> outInt8Block = ubLocal[off_outInt8].template ReinterpretCast<int8_t>();

        // Pre-load gamma
        event_t eventVMTE2G = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2G);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2G);
        DataCopyCustom<T>(x2Block, gammaGm, numCol);
        event_t eventMTE2VG = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2VG);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2VG);

        for (uint32_t i_o = 0; i_o < rowLoop; i_o++) {
            uint32_t curRows = (i_o == rowLoop - 1) ? rowTail : multiRowNum;
            uint32_t curElems = curRows * numCol;
            uint64_t gmBias = static_cast<uint64_t>(i_o) * static_cast<uint64_t>(multiRowNum) *
                              static_cast<uint64_t>(numCol);

            // Stage 1: Add (BF16 → FP32 → BF16)
            DataCopyCustom<T>(x1Block, x1Gm[gmBias], curElems);
            event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V1);

            DataCopyCustom<T>(x2Block, x2Gm[gmBias], curElems);
            event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);

            Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
            Cast(sqxBlock, x2Block, RoundMode::CAST_NONE, curElems);
            PipeBarrier<PIPE_V>();
            Add(xFp32Block, xFp32Block, sqxBlock, curElems);
            PipeBarrier<PIPE_V>();
            Cast(x1Block, xFp32Block, RoundMode::CAST_RINT, curElems);
            PipeBarrier<PIPE_V>();

            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyCustom<T>(yAddGm[gmBias], x1Block, curElems);
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);

            // Stage 2: RmsNorm (FP32)
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            DataCopyCustom<T>(x2Block, gammaGm, numCol);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

            Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
            PipeBarrier<PIPE_V>();

            Mul(sqxBlock, xFp32Block, xFp32Block, curElems);
            PipeBarrier<PIPE_V>();
            Muls(sqxBlock, sqxBlock, avgFactor, curElems);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < curRows; r++) {
                ReduceSumCustom(rstdBlock[r * NUM_PER_BLK_FP32],
                                sqxBlock[r * numCol], tmpBlock, numCol);
            }

            Adds(rstdBlock, rstdBlock, epsilon, curRows * NUM_PER_BLK_FP32);
            PipeBarrier<PIPE_V>();
            Sqrt(rstdBlock, rstdBlock, curRows * NUM_PER_BLK_FP32);
            Duplicate(tmpBlock, ONE_F, NUM_PER_BLK_FP32);
            PipeBarrier<PIPE_V>();

            int32_t repTimes = curRows * NUM_PER_BLK_FP32 / NUM_PER_REP_FP32;
            int32_t tailCnt  = curRows * NUM_PER_BLK_FP32 % NUM_PER_REP_FP32;
            int32_t bodyCnt  = repTimes * NUM_PER_REP_FP32;
            if (likely(repTimes > 0)) {
                Div(rstdBlock, tmpBlock, rstdBlock, NUM_PER_REP_FP32, repTimes,
                    {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
            }
            if (unlikely(tailCnt != 0)) {
                Div(rstdBlock[bodyCnt], tmpBlock, rstdBlock[bodyCnt], tailCnt, 1,
                    {1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE});
            }
            PipeBarrier<PIPE_V>();

            event_t eventVMTE3R = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3R);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3R);
            DataCopyParams rstdCopyParams;
            rstdCopyParams.blockLen = sizeof(float);
            rstdCopyParams.blockCount = curRows;
            DataCopyPad(rstdGm[i_o * multiRowNum], rstdBlock, rstdCopyParams);

            for (uint32_t r = 0; r < curRows; r++) {
                float rstdVal = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
                Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], rstdVal, numCol);
            }
            PipeBarrier<PIPE_V>();

            // Multiply by gamma in FP32 (skip intermediate BF16 round-trip)
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
            Cast(sqxBlock, x2Block, RoundMode::CAST_NONE, numCol);   // gamma BF16 → FP32
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < curRows; r++) {
                Mul(xFp32Block[r * numCol], sqxBlock, xFp32Block[r * numCol], numCol);  // FP32 mul
            }
            PipeBarrier<PIPE_V>();
            Cast(x1Block, xFp32Block, RoundMode::CAST_RINT, curElems);  // FP32 → BF16
            PipeBarrier<PIPE_V>();

            // Stage 3: DynamicQuant (FP32)
            Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
            PipeBarrier<PIPE_V>();

            Abs(sqxBlock, xFp32Block, curElems);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < curRows; r++) {
                ReduceMaxInplace(sqxBlock[r * numCol], numCol);
            }

            for (uint32_t r = 0; r < curRows; r++) {
                float rowMax = sqxBlock.GetValue(r * numCol);
                float scaleVal = rowMax * DYNAMIC_QUANT_INT8_RECIP_SCALE;
                float invScale = (rowMax > DYNAMIC_QUANT_EPSILON) ?
                    (DYNAMIC_QUANT_INT8_SYM_SCALE / rowMax) : 0.0f;
                rstdBlock.SetValue(r * NUM_PER_BLK_FP32, scaleVal);
                Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], invScale, numCol);
            }
            PipeBarrier<PIPE_V>();

            event_t eventVMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3S);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3S);
            DataCopyParams scaleCopyParams;
            scaleCopyParams.blockLen = sizeof(float);
            scaleCopyParams.blockCount = curRows;
            DataCopyPad(scaleGm[i_o * multiRowNum], rstdBlock, scaleCopyParams);

            LocalTensor<int32_t> tmpInt32Block = tmpBlock.template ReinterpretCast<int32_t>();
            LocalTensor<half> tmpHalfBlock = tmpBlock.template ReinterpretCast<half>();
            QuantizeFp32ToInt8(outInt8Block, xFp32Block, tmpInt32Block, tmpHalfBlock, curElems);

            event_t eventVMTE3Q = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Q);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Q);
            DataCopyCustom<int8_t>(yQuantGm[gmBias], outInt8Block, curElems);
        }
    }

private:
    TPipe* pPipe = nullptr;
    TBuf<TPosition::VECCALC> unitBuf;

    // Global tensors
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<int8_t> yQuantGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<T> yAddGm;
    GlobalTensor<float> rstdGm;

    // UB layout sizes (in float elements)
    uint32_t xBlockFloatSz;
    uint32_t fp32BlockSz;
    uint32_t tmpBlockSz;
    uint32_t rstdBlockSz;
    uint32_t outInt8FloatSz;

    // UB layout offsets (in float elements)
    uint32_t off_x1;
    uint32_t off_x2;
    uint32_t off_xFp32;
    uint32_t off_sqx;
    uint32_t off_tmp;
    uint32_t off_rstd;
    uint32_t off_outInt8;

    // Tiling parameters
    uint32_t numRow;
    uint32_t numCol;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
    uint32_t dstType;
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t multiRowNum;
    uint32_t rowWork;
    uint32_t rowLoop;
    uint32_t rowTail;
};

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_AG_MULTI_N_H_
