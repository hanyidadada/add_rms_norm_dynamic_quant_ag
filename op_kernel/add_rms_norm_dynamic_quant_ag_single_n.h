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
 * \file add_rms_norm_dynamic_quant_ag_single_n.h
 * \brief SingleN kernel template for AddRmsNormDynamicQuantAG: process 1 row per iteration per core.
 *        Suitable for large D where UB can only fit 1 row of data.
 *        AG (AllGather) phase appended after computation.
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_AG_SINGLE_N_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_AG_SINGLE_N_H_

#include "add_rms_norm_dynamic_quant_ag_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormDynamicQuantAGSingleN : public KernelAddRmsNormDynamicQuantAGBase {
    static constexpr int32_t MAX_BUFFER = 195584; // (192 - 1) * 1024 bytes
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantAGSingleN(TPipe* pipe)
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

        this->blockIdx_ = GetBlockIdx();

        // Determine rows for this core (head/tail distribution)
        if (this->blockIdx_ < headCoreNum) {
            this->rowWork = this->rowPerHeadCore;
        } else {
            this->rowWork = this->rowPerTailCore;
        }

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
        uint32_t offsetScale = this->rowTotalNum * sizeof(float);
        yQuantGm.SetGlobalBuffer(
            (__gm__ int8_t*)this->buff[this->rankId] + rowOffset * numCol, rowWork * numCol);
        scaleGm.SetGlobalBuffer(
            (__gm__ float*)(this->buff[this->rankId] + offsetScale) + rowOffset, rowWork);

        yAddGm.SetGlobalBuffer((__gm__ T*)yAdd + rowOffset * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + rowOffset, rowWork);

        pPipe->InitBuffer(unitBuf, MAX_BUFFER);
    }

    __aicore__ inline void Process()
    {
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
    // FP16 Path
    // ========================================================================
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
        LocalTensor<int8_t> outInt8Local = ubLocal[ubFactor * 4].template ReinterpretCast<int8_t>();

        for (uint32_t row = 0; row < rowWork; row++) {
            // ================================================================
            // Stage 1: Add (FP16 precision)
            // ================================================================
            // Load x1
            DataCopyCustom<T>(x1Local, x1Gm[row * numCol], numCol);
            event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V1);

            // Load x2
            DataCopyCustom<T>(x2Local, x2Gm[row * numCol], numCol);
            event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);

            // FP16 Add: x1Local = x1 + x2
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();

            // Copy out yAdd (Add result)
            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyCustom<T>(yAddGm[row * numCol], x1Local, numCol);
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);

            // ================================================================
            // Stage 2: RmsNorm (FP32 precision)
            // ================================================================

            // Copy gamma into x2Local (reuse buffer)
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            DataCopyCustom<T>(x2Local, gammaGm, numCol);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

            // Cast FP16 → FP32
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // sqx = x^2
            Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
            PipeBarrier<PIPE_V>();

            // sqx = sqx * avgFactor (mean of squares)
            Muls(sqxLocal, sqxLocal, avgFactor, numCol);
            PipeBarrier<PIPE_V>();

            // Reduce sum: sqx → single value
            ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
            PipeBarrier<PIPE_V>();

            // rstd = 1 / sqrt(mean_sq + epsilon)
            Adds(sqxLocal, sqxLocal, epsilon, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(sqxLocal, sqxLocal, 1);
            Duplicate(tmpLocal, ONE_F, 1);
            PipeBarrier<PIPE_V>();
            Div(sqxLocal, tmpLocal, sqxLocal, 1);
            PipeBarrier<PIPE_V>();

            // Copy out rstd
            event_t eventVMTE3Rstd = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Rstd);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Rstd);
            DataCopyParams rstdCopyParams;
            rstdCopyParams.blockLen = sizeof(float);
            rstdCopyParams.blockCount = 1;
            DataCopyPad(rstdGm[row], sqxLocal, rstdCopyParams);

            // Extract rstd scalar value
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdValue = sqxLocal.GetValue(0);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);

            // x_norm = x * rstd
            Muls(xFp32Local, xFp32Local, rstdValue, numCol);
            PipeBarrier<PIPE_V>();

            // Cast FP32 → FP16 (intermediate RmsNorm result, stays in UB)
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
            Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // Multiply by gamma (FP16)
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
            Mul(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();

            // ================================================================
            // Stage 3: DynamicQuant (FP32 precision)
            // ================================================================

            // Cast FP16 → FP32 (quant input, stays in UB - fusion key!)
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // abs(xFp32) into sqxLocal
            Abs(sqxLocal, xFp32Local, numCol);
            PipeBarrier<PIPE_V>();

            // Reduce max in-place: find row-wise max(abs) in sqxLocal[0]
            ReduceMaxInplace(sqxLocal, numCol);

            // Extract max value
            event_t eventVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS2);
            WaitFlag<HardEvent::V_S>(eventVS2);
            float rowMax = sqxLocal.GetValue(0);
            event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV2);
            WaitFlag<HardEvent::S_V>(eventSV2);

            // scale = rowMax / 127.0
            float scaleVal = rowMax * DYNAMIC_QUANT_INT8_RECIP_SCALE;

            // invScale = 127.0 / rowMax (with epsilon guard)
            float invScale = (rowMax > DYNAMIC_QUANT_EPSILON) ?
                (DYNAMIC_QUANT_INT8_SYM_SCALE / rowMax) : 0.0f;

            // Copy out scale
            sqxLocal.SetValue(0, scaleVal);
            PipeBarrier<PIPE_V>();
            event_t eventVMTE3Scale = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
            DataCopyParams scaleCopyParams;
            scaleCopyParams.blockLen = sizeof(float);
            scaleCopyParams.blockCount = 1;
            DataCopyPad(scaleGm[row], sqxLocal, scaleCopyParams);

            // xFp32 = xFp32 * invScale
            Muls(xFp32Local, xFp32Local, invScale, numCol);
            PipeBarrier<PIPE_V>();

            // Quantize: FP32 → INT32 (round) → FP16 (round) → INT8 (trunc)
            LocalTensor<int32_t> tmpInt32Local = tmpLocal.template ReinterpretCast<int32_t>();
            LocalTensor<half> tmpHalfLocal = tmpLocal.template ReinterpretCast<half>();
            QuantizeFp32ToInt8(outInt8Local, xFp32Local, tmpInt32Local, tmpHalfLocal, numCol);

            // Copy out yQuant
            event_t eventVMTE3Quant = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
            DataCopyCustom<int8_t>(yQuantGm[row * numCol], outInt8Local, numCol);
        }
    }

    // ========================================================================
    // BF16 Path
    // ========================================================================
    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = unitBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
        LocalTensor<int8_t> outInt8Local = ubLocal[ubFactor * 4].template ReinterpretCast<int8_t>();

        for (uint32_t row = 0; row < rowWork; row++) {
            // ================================================================
            // Stage 1: Add (BF16 → FP32 → BF16 path)
            // ================================================================

            // Load x1
            DataCopyCustom<T>(x1Local, x1Gm[row * numCol], numCol);
            event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V1);

            // Load x2
            DataCopyCustom<T>(x2Local, x2Gm[row * numCol], numCol);
            event_t eventMTE2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);

            // BF16 does not support direct Add: cast both to FP32 first
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();
            // Cast FP32 → BF16 for output
            Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();

            // Copy out yAdd (Add result)
            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            DataCopyCustom<T>(yAddGm[row * numCol], x1Local, numCol);
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);

            // ================================================================
            // Stage 2: RmsNorm (FP32 precision)
            // ================================================================

            // Copy gamma
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            DataCopyCustom<T>(x2Local, gammaGm, numCol);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V2);

            // Cast BF16 → FP32 for RMS computation
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // sqx = x^2
            Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
            PipeBarrier<PIPE_V>();

            // sqx = sqx * avgFactor
            Muls(sqxLocal, sqxLocal, avgFactor, numCol);
            PipeBarrier<PIPE_V>();

            // Reduce sum
            ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
            PipeBarrier<PIPE_V>();

            // rstd = 1 / sqrt(mean_sq + epsilon)
            Adds(sqxLocal, sqxLocal, epsilon, 1);
            PipeBarrier<PIPE_V>();
            Sqrt(sqxLocal, sqxLocal, 1);
            Duplicate(tmpLocal, ONE_F, 1);
            PipeBarrier<PIPE_V>();
            Div(sqxLocal, tmpLocal, sqxLocal, 1);
            PipeBarrier<PIPE_V>();

            // Copy out rstd
            event_t eventVMTE3Rstd = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Rstd);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Rstd);
            DataCopyParams rstdCopyParams;
            rstdCopyParams.blockLen = sizeof(float);
            rstdCopyParams.blockCount = 1;
            DataCopyPad(rstdGm[row], sqxLocal, rstdCopyParams);

            // Extract rstd scalar
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float rstdValue = sqxLocal.GetValue(0);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);

            // x_norm = x * rstd
            Muls(xFp32Local, xFp32Local, rstdValue, numCol);
            PipeBarrier<PIPE_V>();

            // Cast FP32 → BF16
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
            Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();

            // Cast BF16 → FP32 for gamma multiplication (BF16 * BF16 not directly supported)
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // Load gamma into FP32
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V2);
            Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // Multiply by gamma (FP32)
            Mul(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();

            // Cast back to BF16
            Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();

            // ================================================================
            // Stage 3: DynamicQuant (FP32 precision)
            // ================================================================

            // Cast BF16 → FP32
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();

            // abs(xFp32)
            Abs(sqxLocal, xFp32Local, numCol);
            PipeBarrier<PIPE_V>();

            // Reduce max
            ReduceMaxInplace(sqxLocal, numCol);

            // Extract max
            event_t eventVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS2);
            WaitFlag<HardEvent::V_S>(eventVS2);
            float rowMax = sqxLocal.GetValue(0);
            event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV2);
            WaitFlag<HardEvent::S_V>(eventSV2);

            float scaleVal = rowMax * DYNAMIC_QUANT_INT8_RECIP_SCALE;
            float invScale = (rowMax > DYNAMIC_QUANT_EPSILON) ?
                (DYNAMIC_QUANT_INT8_SYM_SCALE / rowMax) : 0.0f;

            // Copy out scale
            sqxLocal.SetValue(0, scaleVal);
            PipeBarrier<PIPE_V>();
            event_t eventVMTE3Scale = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
            DataCopyParams scaleCopyParams;
            scaleCopyParams.blockLen = sizeof(float);
            scaleCopyParams.blockCount = 1;
            DataCopyPad(scaleGm[row], sqxLocal, scaleCopyParams);

            // xFp32 = xFp32 * invScale
            Muls(xFp32Local, xFp32Local, invScale, numCol);
            PipeBarrier<PIPE_V>();

            // Quantize
            LocalTensor<int32_t> tmpInt32Local = tmpLocal.template ReinterpretCast<int32_t>();
            LocalTensor<half> tmpHalfLocal = tmpLocal.template ReinterpretCast<half>();
            QuantizeFp32ToInt8(outInt8Local, xFp32Local, tmpInt32Local, tmpHalfLocal, numCol);

            // Copy out yQuant
            event_t eventVMTE3Quant = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
            DataCopyCustom<int8_t>(yQuantGm[row * numCol], outInt8Local, numCol);
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
    uint32_t rowWork;
};

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_AG_SINGLE_N_H_
