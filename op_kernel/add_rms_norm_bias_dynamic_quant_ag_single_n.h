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
 * \file add_rms_norm_bias_dynamic_quant_ag_single_n.h
 * \brief SingleN kernel for AddRmsNormBiasDynamicQuantAG: 1 row per iteration.
 *
 * 3-stage compute pipeline per row:
 *   Stage 1 (Add):        x = x1 + x2, copy out x to GM
 *   Stage 2 (RmsNorm):    y = x * rstd * gamma, copy out y and rstd to GM
 *   Stage 3 (DynamicQuant): y_quant = quantize(y), copy out scale and y_quant to HCCL window
 * Followed by AG phase: AllGather via MC2.
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_SINGLE_N_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_SINGLE_N_H_

#include "add_rms_norm_bias_dynamic_quant_ag_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormBiasDynamicQuantAGSingleN : public KernelAddRmsNormBiasDynamicQuantAGBase {
    static constexpr int32_t MAX_BUFFER = 195584; // (192 - 1) * 1024 bytes
public:
    __aicore__ inline KernelAddRmsNormBiasDynamicQuantAGSingleN(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR bias,
        GM_ADDR yQuant, GM_ADDR scale, GM_ADDR x,
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
        this->hasX2 = tiling->hasX2;
        this->hasBias = tiling->hasBias;

        this->blockIdx_ = GetBlockIdx();

        // Determine rows for this core (head/tail distribution)
        if (this->blockIdx_ < headCoreNum) {
            this->rowWork = this->rowPerHeadCore;
        } else {
            this->rowWork = this->rowPerTailCore;
        }

        // Guard: cores beyond useCoreNum are AG-only (groupSize > useCoreNum)
        if (this->blockIdx_ >= tiling->coreNum) {
            this->rowWork = 0;
        }

        this->rowWork_ = this->rowWork;

        // Guard: extra cores launched for AG only
        if (this->rowWork == 0) {
            this->InitAGParams(tiling);
            this->y1Out = yQuant;
            this->scale1Out = scale;
            pPipe->InitBuffer(unitBuf, MAX_BUFFER);
            return;
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
        if (this->hasX2) {
            x2Gm.SetGlobalBuffer((__gm__ T*)x2 + rowOffset * numCol, rowWork * numCol);
        }
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if (this->hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T*)bias, numCol);
        }

        // Quantized output redirected to HCCL window
        uint32_t offsetScale = this->rowTotalNum * this->rowLen;
        yQuantGm.SetGlobalBuffer(
            (__gm__ int8_t*)this->buff[this->rankId] + rowOffset * numCol, rowWork * numCol);
        scaleGm.SetGlobalBuffer(
            (__gm__ float*)(this->buff[this->rankId] + offsetScale) + rowOffset, rowWork);

        // Intermediate output: x (add result)
        xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * numCol, rowWork * numCol);

        pPipe->InitBuffer(unitBuf, MAX_BUFFER);
    }

    __aicore__ inline void Process()
    {
        if (this->rowWork_ == 0) {
            // Extra core: skip computation, AG sync only
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
    // FP16 Path — matches standalone add_rms_norm_bias + dynamic_quant calling order
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
            // Stage 1: Add (x = x1 + x2, FP16 precision)
            // ================================================================
            StageAddFp16(row, x1Local, x2Local);

            // ================================================================
            // Stage 2: RmsNorm (y = x * rstd * gamma, FP32 internal)
            // ================================================================
            StageRmsNormFp16(row, x1Local, x2Local, xFp32Local, sqxLocal, tmpLocal);

            // ================================================================
            // Stage 3: DynamicQuant (y_quant = round(y * invScale), FP32 internal)
            // ================================================================
            StageDynamicQuantFp16(row, x1Local, xFp32Local, sqxLocal, tmpLocal, outInt8Local);
        }
    }

    // ---- Stage 1: Add (FP16) ----
    __aicore__ inline void StageAddFp16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local)
    {
        // Load x1
        DataCopyCustom<T>(x1Local, x1Gm[row * numCol], numCol);
        PipeBarrier<PIPE_V>();

        if (this->hasX2) {
            // Load x2
            DataCopyCustom<T>(x2Local, x2Gm[row * numCol], numCol);
            PipeBarrier<PIPE_V>();

            // FP16 Add: x1Local = x1 + x2
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
        }

        // Copy out x (add result)
        DataCopyCustom<T>(xGm[row * numCol], x1Local, numCol);
        PipeBarrier<PIPE_V>();
    }

    // ---- Stage 2: RmsNorm (FP16) ----
    __aicore__ inline void StageRmsNormFp16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal)
    {
        // Copy gamma into x2Local (reuse buffer)
        DataCopyCustom<T>(x2Local, gammaGm, numCol);
        PipeBarrier<PIPE_V>();

        // Cast FP16 → FP32
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // sqx = x^2
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();

        // sqx *= avgFactor (1/numCol)
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        // Reduce sum → single value
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();

        // rstd = 1 / sqrt(sum + epsilon)
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE_F, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // Extract rstd scalar
        float rstdValue = sqxLocal.GetValue(0);
        PipeBarrier<PIPE_V>();

        // x_norm = x * rstd
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        // Cast FP32 → FP16
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // Multiply by gamma (FP16)
        Mul(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();

        // Add bias if provided (FP16)
        if (this->hasBias) {
            DataCopyCustom<T>(x2Local, biasGm, numCol);
            PipeBarrier<PIPE_V>();
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
        }
    }

    // ---- Stage 3: DynamicQuant (FP16) ----
    __aicore__ inline void StageDynamicQuantFp16(uint32_t row,
        LocalTensor<T>& x1Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal,
        LocalTensor<int8_t>& outInt8Local)
    {
        // Cast FP16 → FP32 (quant input, stays in UB — fusion key!)
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // abs(xFp32)
        Abs(sqxLocal, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();

        // Reduce max in-place
        ReduceMaxInplace(sqxLocal, numCol);
        PipeBarrier<PIPE_V>();

        // invScale = 127.0 / rowMax
        LocalTensor<float> constScale = tmpLocal;
        Duplicate<float>(constScale, DYNAMIC_QUANT_INT8_SYM_SCALE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, constScale, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // Extract invScale scalar
        event_t eventVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS2);
        WaitFlag<HardEvent::V_S>(eventVS2);
        float invScale = sqxLocal.GetValue(0);
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);

        // scaleOut = max_abs / 127.0
        float scaleVal = 1.0f / invScale;

        // Copy out scale to HCCL window
        sqxLocal.SetValue(0, scaleVal);
        PipeBarrier<PIPE_V>();
        event_t eventVMTE3Scale = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
        DataCopyParams scaleCopyParams;
        scaleCopyParams.blockLen = sizeof(float);
        scaleCopyParams.blockCount = 1;
        DataCopyPad(scaleGm[row], sqxLocal, scaleCopyParams);

        // xFp32 *= invScale
        Muls(xFp32Local, xFp32Local, invScale, numCol);
        PipeBarrier<PIPE_V>();

        // Quantize: FP32 → INT16(round) → half(round) → INT8(trunc)
        LocalTensor<int32_t> tmpInt32Local = tmpLocal.template ReinterpretCast<int32_t>();
        LocalTensor<half> tmpHalfLocal = tmpLocal.template ReinterpretCast<half>();
        QuantizeFp32ToInt8(outInt8Local, xFp32Local, tmpInt32Local, tmpHalfLocal, numCol);

        // Copy out yQuant to HCCL window
        event_t eventVMTE3Quant = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
        DataCopyCustom<int8_t>(yQuantGm[row * numCol], outInt8Local, numCol);
    }

    // ========================================================================
    // BF16 Path — matches standalone add_rms_norm_bias + dynamic_quant calling order
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
            StageAddBf16(row, x1Local, x2Local, xFp32Local, sqxLocal);
            StageRmsNormBf16(row, x1Local, x2Local, xFp32Local, sqxLocal, tmpLocal);
            StageDynamicQuantBf16(row, x1Local, xFp32Local, sqxLocal, tmpLocal, outInt8Local);
        }
    }

    // ---- Stage 1: Add (BF16) ----
    __aicore__ inline void StageAddBf16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal)
    {
        // Load x1
        DataCopyCustom<T>(x1Local, x1Gm[row * numCol], numCol);
        PipeBarrier<PIPE_V>();

        if (this->hasX2) {
            // Load x2
            DataCopyCustom<T>(x2Local, x2Gm[row * numCol], numCol);
            PipeBarrier<PIPE_V>();

            // BF16: cast both to FP32, add, cast back to BF16
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();
            Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        }

        // Reload BF16-rounded values for rmsnorm consistency (match standalone kernel precision)
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // Copy out x (add result)
        DataCopyCustom<T>(xGm[row * numCol], x1Local, numCol);
        PipeBarrier<PIPE_V>();
    }

    // ---- Stage 2: RmsNorm (BF16) ----
    __aicore__ inline void StageRmsNormBf16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal)
    {
        // Copy gamma
        DataCopyCustom<T>(x2Local, gammaGm, numCol);
        PipeBarrier<PIPE_V>();

        // xFp32Local holds BF16-rounded sum from StageAddBf16 — matches standalone kernel precision

        // sqx = x^2
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();

        // sqx *= avgFactor
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();

        // Reduce sum
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();

        // rstd = 1 / sqrt(sum + epsilon)
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE_F, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // Extract rstd scalar
        float rstdValue = sqxLocal.GetValue(0);
        PipeBarrier<PIPE_V>();

        // x_norm = x * rstd
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        // Cast FP32 → BF16
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();

        // Cast BF16 → FP32 for gamma multiply (BF16 * BF16 not supported)
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // Load gamma into FP32
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // Multiply by gamma (FP32)
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol);
        PipeBarrier<PIPE_V>();

        // Add bias if provided (FP32)
        if (this->hasBias) {
            DataCopyCustom<T>(x2Local, biasGm, numCol);
            PipeBarrier<PIPE_V>();
            Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();
        }

        // Cast back to BF16
        PipeBarrier<PIPE_V>();
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
    }

    // ---- Stage 3: DynamicQuant (BF16) ----
    __aicore__ inline void StageDynamicQuantBf16(uint32_t row,
        LocalTensor<T>& x1Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal,
        LocalTensor<int8_t>& outInt8Local)
    {
        // Cast BF16 → FP32
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // abs(xFp32)
        Abs(sqxLocal, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();

        // Reduce max
        ReduceMaxInplace(sqxLocal, numCol);
        PipeBarrier<PIPE_V>();

        // invScale = 127.0 / rowMax
        LocalTensor<float> constScaleBf16 = tmpLocal;
        Duplicate<float>(constScaleBf16, DYNAMIC_QUANT_INT8_SYM_SCALE, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, constScaleBf16, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // Extract invScale scalar
        event_t eventVS2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS2);
        WaitFlag<HardEvent::V_S>(eventVS2);
        float invScale = sqxLocal.GetValue(0);
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);

        // scaleOut = max_abs / 127.0
        float scaleVal = 1.0f / invScale;

        // Copy out scale to HCCL window
        sqxLocal.SetValue(0, scaleVal);
        PipeBarrier<PIPE_V>();
        event_t eventVMTE3Scale = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Scale);
        DataCopyParams scaleCopyParams;
        scaleCopyParams.blockLen = sizeof(float);
        scaleCopyParams.blockCount = 1;
        DataCopyPad(scaleGm[row], sqxLocal, scaleCopyParams);

        // xFp32 *= invScale
        Muls(xFp32Local, xFp32Local, invScale, numCol);
        PipeBarrier<PIPE_V>();

        // Quantize: FP32 → INT16(round) → half(round) → INT8(trunc)
        LocalTensor<int32_t> tmpInt32Local = tmpLocal.template ReinterpretCast<int32_t>();
        LocalTensor<half> tmpHalfLocal = tmpLocal.template ReinterpretCast<half>();
        QuantizeFp32ToInt8(outInt8Local, xFp32Local, tmpInt32Local, tmpHalfLocal, numCol);

        // Copy out yQuant to HCCL window
        event_t eventVMTE3Quant = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3Quant);
        DataCopyCustom<int8_t>(yQuantGm[row * numCol], outInt8Local, numCol);
    }

private:
    TPipe* pPipe = nullptr;
    TBuf<TPosition::VECCALC> unitBuf;

    // Global tensors
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<int8_t> yQuantGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<T> xGm;       // add result

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

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_SINGLE_N_H_
