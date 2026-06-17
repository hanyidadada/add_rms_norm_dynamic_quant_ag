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
 * \file add_rms_norm_bias_dynamic_quant_ag_multi_n.h
 * \brief MultiN kernel for AddRmsNormBiasDynamicQuantAG: multiple rows per iteration.
 *
 * 3-stage compute pipeline per row-group:
 *   Stage 1 (Add):        x = x1 + x2, copy out x to GM
 *   Stage 2 (RmsNorm):    y = x * rstd * gamma, copy out y and rstd to GM
 *   Stage 3 (DynamicQuant): y_quant = quantize(y), copy out scale and y_quant to HCCL window
 * Followed by AG phase.
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_MULTI_N_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_MULTI_N_H_

#include "add_rms_norm_bias_dynamic_quant_ag_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormBiasDynamicQuantAGMultiN : public KernelAddRmsNormBiasDynamicQuantAGBase {
    static constexpr int32_t MAX_BUFFER = 195584;
    static constexpr uint32_t SZ_FLOAT = sizeof(float);    // 4
    static constexpr uint32_t SZ_HALF  = sizeof(half);     // 2
    static constexpr uint32_t SZ_INT8  = sizeof(int8_t);   // 1
    static constexpr uint32_t T_SZ = sizeof(T);            // 2 for half/bf16

public:
    __aicore__ inline KernelAddRmsNormBiasDynamicQuantAGMultiN(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
        GM_ADDR yQuant, GM_ADDR scale, GM_ADDR x, GM_ADDR y, GM_ADDR rstd,
        GM_ADDR workspace, const AddRmsNormBiasDynamicQuantAGTilingData* tiling)
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

        if (this->blockIdx_ < headCoreNum) {
            this->rowWork = this->rowPerHeadCore;
        } else {
            this->rowWork = this->rowPerTailCore;
        }

        if (this->blockIdx_ >= tiling->coreNum) {
            this->rowWork = 0;
        }

        this->rowWork_ = this->rowWork;

        if (this->rowWork == 0) {
            this->InitAGParams(tiling);
            this->y1Out = yQuant;
            this->scale1Out = scale;
            pPipe->InitBuffer(unitBuf, MAX_BUFFER);
            return;
        }

        this->rowLoop = CeilDiv(this->rowWork, this->multiRowNum);
        this->rowTail = this->rowWork - (this->rowLoop - 1) * this->multiRowNum;

        uint32_t rowOffset = 0;
        if (this->blockIdx_ < headCoreNum) {
            rowOffset = this->blockIdx_ * rowPerHeadCore;
        } else {
            rowOffset = headCoreNum * rowPerHeadCore + (this->blockIdx_ - headCoreNum) * rowPerTailCore;
        }

        this->InitAGParams(tiling);

        this->y1Out = yQuant;
        this->scale1Out = scale;

        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + rowOffset * numCol, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + rowOffset * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);

        uint32_t offsetScale = this->rowTotalNum * this->rowLen;
        yQuantGm.SetGlobalBuffer(
            (__gm__ int8_t*)this->buff[this->rankId] + rowOffset * numCol, rowWork * numCol);
        scaleGm.SetGlobalBuffer(
            (__gm__ float*)(this->buff[this->rankId] + offsetScale) + rowOffset, rowWork);

        xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * numCol, rowWork * numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + rowOffset * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + rowOffset, rowWork);

        ComputeUBLayout();
        pPipe->InitBuffer(unitBuf, MAX_BUFFER);
    }

    __aicore__ inline void Process()
    {
        if (this->rowWork_ == 0) {
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

        PipeBarrier<PIPE_ALL>();
        pPipe->Reset();
        pPipe->InitBuffer(this->copyBuf, USED_UB_SIZE);
        pPipe->InitBuffer(this->flagBuf, 32);
        this->ProcessAG();
    }

private:
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

            // ---- Stage 1: Add (FP16) ----
            StageAddFp16(gmBias, curElems, x1Block, x2Block);

            // ---- Stage 2: RmsNorm (FP16) ----
            StageRmsNormFp16(i_o, gmBias, curRows, curElems,
                x1Block, x2Block, xFp32Block, sqxBlock, tmpBlock, rstdBlock);

            // ---- Stage 3: DynamicQuant (FP16) ----
            StageDynamicQuantFp16(i_o, gmBias, curRows, curElems,
                x1Block, xFp32Block, sqxBlock, tmpBlock, rstdBlock, outInt8Block);
        }
    }

    // ---- Stage 1: Add (FP16 MultiN) ----
    __aicore__ inline void StageAddFp16(uint64_t gmBias, uint32_t curElems,
        LocalTensor<T>& x1Block, LocalTensor<T>& x2Block)
    {
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

        // Copy out x (add result)
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(xGm[gmBias], x1Block, curElems);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
    }

    // ---- Stage 2: RmsNorm (FP16 MultiN) ----
    __aicore__ inline void StageRmsNormFp16(uint32_t i_o, uint64_t gmBias,
        uint32_t curRows, uint32_t curElems,
        LocalTensor<T>& x1Block, LocalTensor<T>& x2Block,
        LocalTensor<float>& xFp32Block, LocalTensor<float>& sqxBlock,
        LocalTensor<float>& tmpBlock, LocalTensor<float>& rstdBlock)
    {
        // Re-load gamma (x2Block was overwritten by Add stage)
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        DataCopyCustom<T>(x2Block, gammaGm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
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

        if (curRows == 1) {
            Div(rstdBlock, tmpBlock, rstdBlock, 1);
        } else {
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
        }
        PipeBarrier<PIPE_V>();

        // Copy out rstd
        event_t eventVMTE3R = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3R);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3R);

        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdVal = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], rstdVal, numCol);
        }
        PipeBarrier<PIPE_V>();

        // Compact rstd from stride-8 to stride-1 before GM copy
        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdVal = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            tmpBlock.SetValue(r, rstdVal);
        }
        PipeBarrier<PIPE_V>();
        DataCopyParams rstdCopyParams;
        rstdCopyParams.blockLen = sizeof(float);
        rstdCopyParams.blockCount = curRows;
        DataCopyPad(rstdGm[i_o * multiRowNum], tmpBlock, rstdCopyParams);

        // Wait for x copy-out to finish, then cast FP32→FP16 for gamma multiply
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Cast(x1Block, xFp32Block, RoundMode::CAST_NONE, curElems);
        PipeBarrier<PIPE_V>();

        // Multiply by gamma (FP16)
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        for (uint32_t r = 0; r < curRows; r++) {
            Mul(x1Block[r * numCol], x2Block, x1Block[r * numCol], numCol);
        }
        PipeBarrier<PIPE_V>();

        // Copy out y (rmsnorm result) — before Cast to FP32 for quant destroys x1Block
        event_t eventVMTE3Y = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Y);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Y);
        DataCopyCustom<T>(yGm[gmBias], x1Block, curElems);
    }

    // ---- Stage 3: DynamicQuant (FP16 MultiN) ----
    __aicore__ inline void StageDynamicQuantFp16(uint32_t i_o, uint64_t gmBias,
        uint32_t curRows, uint32_t curElems,
        LocalTensor<T>& x1Block,
        LocalTensor<float>& xFp32Block, LocalTensor<float>& sqxBlock,
        LocalTensor<float>& tmpBlock, LocalTensor<float>& rstdBlock,
        LocalTensor<int8_t>& outInt8Block)
    {
        // Cast FP16 → FP32
        Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
        PipeBarrier<PIPE_V>();

        Abs(sqxBlock, xFp32Block, curElems);
        PipeBarrier<PIPE_V>();

        for (uint32_t r = 0; r < curRows; r++) {
            ReduceMaxInplace(sqxBlock[r * numCol], numCol);
        }
        PipeBarrier<PIPE_V>();

        // invScale = 127.0 / rowMax
        LocalTensor<float> constScale = tmpBlock;
        Duplicate<float>(constScale, DYNAMIC_QUANT_INT8_SYM_SCALE, NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < curRows; r++) {
            Div(sqxBlock[r * numCol], constScale, sqxBlock[r * numCol], 1);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float invScale = sqxBlock.GetValue(r * numCol);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            float scaleVal = 1.0f / invScale;
            rstdBlock.SetValue(r * NUM_PER_BLK_FP32, scaleVal);
            Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], invScale, numCol);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> tmpInt32Block = tmpBlock.template ReinterpretCast<int32_t>();
        LocalTensor<half> tmpHalfBlock = tmpBlock.template ReinterpretCast<half>();
        QuantizeFp32ToInt8(outInt8Block, xFp32Block, tmpInt32Block, tmpHalfBlock, curElems);

        // Compact scale values from stride-8 to stride-1 before GM copy
        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float sv = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            tmpBlock.SetValue(r, sv);
        }
        PipeBarrier<PIPE_V>();

        // Copy out scale to HCCL window
        event_t eventVMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3S);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3S);
        DataCopyParams scaleCopyParams;
        scaleCopyParams.blockLen = sizeof(float);
        scaleCopyParams.blockCount = curRows;
        DataCopyPad(scaleGm[i_o * multiRowNum], tmpBlock, scaleCopyParams);

        // Copy out yQuant to HCCL window
        event_t eventVMTE3Q = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Q);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Q);
        DataCopyCustom<int8_t>(yQuantGm[gmBias], outInt8Block, curElems);
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

            // ---- Stage 1: Add (BF16) ----
            StageAddBf16(gmBias, curElems, x1Block, x2Block, xFp32Block, sqxBlock);

            // ---- Stage 2: RmsNorm (BF16) ----
            StageRmsNormBf16(i_o, gmBias, curRows, curElems,
                x1Block, x2Block, xFp32Block, sqxBlock, tmpBlock, rstdBlock);

            // ---- Stage 3: DynamicQuant (BF16) ----
            StageDynamicQuantBf16(i_o, gmBias, curRows, curElems,
                x1Block, xFp32Block, sqxBlock, tmpBlock, rstdBlock, outInt8Block);
        }
    }

    // ---- Stage 1: Add (BF16 MultiN) ----
    __aicore__ inline void StageAddBf16(uint64_t gmBias, uint32_t curElems,
        LocalTensor<T>& x1Block, LocalTensor<T>& x2Block,
        LocalTensor<float>& xFp32Block, LocalTensor<float>& sqxBlock)
    {
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

        // Copy out x (add result)
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyCustom<T>(xGm[gmBias], x1Block, curElems);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
    }

    // ---- Stage 2: RmsNorm (BF16 MultiN) ----
    __aicore__ inline void StageRmsNormBf16(uint32_t i_o, uint64_t gmBias,
        uint32_t curRows, uint32_t curElems,
        LocalTensor<T>& x1Block, LocalTensor<T>& x2Block,
        LocalTensor<float>& xFp32Block, LocalTensor<float>& sqxBlock,
        LocalTensor<float>& tmpBlock, LocalTensor<float>& rstdBlock)
    {
        // Re-load gamma
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        DataCopyCustom<T>(x2Block, gammaGm, numCol);
        event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

        // xFp32Block still holds FP32 sum from Add — no re-cast needed

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

        if (curRows == 1) {
            Div(rstdBlock, tmpBlock, rstdBlock, 1);
        } else {
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
        }
        PipeBarrier<PIPE_V>();

        // Copy out rstd
        event_t eventVMTE3R = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3R);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3R);

        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdVal = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], rstdVal, numCol);
        }
        PipeBarrier<PIPE_V>();

        // Compact rstd from stride-8 to stride-1
        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdVal = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            tmpBlock.SetValue(r, rstdVal);
        }
        PipeBarrier<PIPE_V>();
        DataCopyParams rstdCopyParams;
        rstdCopyParams.blockLen = sizeof(float);
        rstdCopyParams.blockCount = curRows;
        DataCopyPad(rstdGm[i_o * multiRowNum], tmpBlock, rstdCopyParams);

        // Cast FP32→BF16, then FP32 for gamma multiply
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        Cast(x1Block, xFp32Block, RoundMode::CAST_RINT, curElems);
        PipeBarrier<PIPE_V>();

        WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
        Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
        PipeBarrier<PIPE_V>();
        Cast(sqxBlock, x2Block, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < curRows; r++) {
            Mul(xFp32Block[r * numCol], sqxBlock, xFp32Block[r * numCol], numCol);
        }
        PipeBarrier<PIPE_V>();
        Cast(x1Block, xFp32Block, RoundMode::CAST_RINT, curElems);
        PipeBarrier<PIPE_V>();

        // Copy out y (rmsnorm result)
        event_t eventVMTE3Y = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Y);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Y);
        DataCopyCustom<T>(yGm[gmBias], x1Block, curElems);
    }

    // ---- Stage 3: DynamicQuant (BF16 MultiN) ----
    __aicore__ inline void StageDynamicQuantBf16(uint32_t i_o, uint64_t gmBias,
        uint32_t curRows, uint32_t curElems,
        LocalTensor<T>& x1Block,
        LocalTensor<float>& xFp32Block, LocalTensor<float>& sqxBlock,
        LocalTensor<float>& tmpBlock, LocalTensor<float>& rstdBlock,
        LocalTensor<int8_t>& outInt8Block)
    {
        // Cast BF16 → FP32
        Cast(xFp32Block, x1Block, RoundMode::CAST_NONE, curElems);
        PipeBarrier<PIPE_V>();

        Abs(sqxBlock, xFp32Block, curElems);
        PipeBarrier<PIPE_V>();

        for (uint32_t r = 0; r < curRows; r++) {
            ReduceMaxInplace(sqxBlock[r * numCol], numCol);
        }
        PipeBarrier<PIPE_V>();

        // invScale = 127.0 / rowMax
        LocalTensor<float> constScaleBf16 = tmpBlock;
        Duplicate<float>(constScaleBf16, DYNAMIC_QUANT_INT8_SYM_SCALE, NUM_PER_BLK_FP32);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < curRows; r++) {
            Div(sqxBlock[r * numCol], constScaleBf16, sqxBlock[r * numCol], 1);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float invScale = sqxBlock.GetValue(r * numCol);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            float scaleVal = 1.0f / invScale;
            rstdBlock.SetValue(r * NUM_PER_BLK_FP32, scaleVal);
            Muls(xFp32Block[r * numCol], xFp32Block[r * numCol], invScale, numCol);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> tmpInt32Block = tmpBlock.template ReinterpretCast<int32_t>();
        LocalTensor<half> tmpHalfBlock = tmpBlock.template ReinterpretCast<half>();
        QuantizeFp32ToInt8(outInt8Block, xFp32Block, tmpInt32Block, tmpHalfBlock, curElems);

        // Compact scale values from stride-8 to stride-1
        for (uint32_t r = 0; r < curRows; r++) {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float sv = rstdBlock.GetValue(r * NUM_PER_BLK_FP32);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            tmpBlock.SetValue(r, sv);
        }
        PipeBarrier<PIPE_V>();

        event_t eventVMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3S);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3S);
        DataCopyParams scaleCopyParams;
        scaleCopyParams.blockLen = sizeof(float);
        scaleCopyParams.blockCount = curRows;
        DataCopyPad(scaleGm[i_o * multiRowNum], tmpBlock, scaleCopyParams);

        event_t eventVMTE3Q = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Q);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Q);
        DataCopyCustom<int8_t>(yQuantGm[gmBias], outInt8Block, curElems);
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
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
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

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_MULTI_N_H_
