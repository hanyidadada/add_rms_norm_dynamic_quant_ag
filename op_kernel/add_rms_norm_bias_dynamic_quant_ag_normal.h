/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_normal.h
 * \brief Normal+db kernel for AddRmsNormBiasDynamicQuantAG (TilingKey 1003/3002).
 *
 * ============ 设计要点 ============
 * 1. UB 内存布局 (4 个独立 TBuf, 总量由 host rowBatchSize 公式保证 <= ubSize):
 *    - computeBuf : 单行计算工作集 (每行复用)  = x1|x2|xFp32|sqx|tmp|constArea
 *    - xStageBuf  : rowBatchSize 行 x 暂存 (T)
 *    - yQuantStageBuf : rowBatchSize 行 yQuant 暂存 (int8, 行步 outAlignLen)
 *    - scaleStageBuf  : rowBatchSize 个 scale 暂存 (float, 8 对齐)
 * 2. 逐行计算 add_rms_norm_bias + dynamic_quant, 结果写入对应暂存行的 batchOffset 槽:
 *    - StageAdd        : x = x1 + x2  -> xStage[b]; 同时保留 fp32 和供 rmsnorm
 *    - StageRmsNorm    : y = (x*rstd)*gamma + bias -> x1Local (T 精度 y)
 *    - StageDynamicQuant : absmax(y), scale=absmax/127, yQuant=trunc(round(y/scale))
 *                         -> yQuantStage[b], scaleStage[b]
 * 3. 聚合写 GM: 每处理完 rowBatchSize 行 (或末尾不足一批), 发 3 次批量 DataCopyPad
 *    (x / yQuant / scale), 严禁逐行写 GM。
 *
 * ============ API 对齐 (与单算子逐调用一致) ============
 *  - rmsnorm 阶段: 复刻 add_rms_norm_bias.h 的 CopyIn + Compute<half/bfloat16_t> (normal 分支)。
 *    * BF16 用 fp32 求和(未取整)做 rmsnorm —— 不做旧融合算子 normal.h:406 的 bf16 reload。
 *  - dynamic_quant 阶段: 复刻 dynamic_quant_db.h 的 ComputeRowMax + QuantizeRow (db 对称分支)。
 *    * 向量化 Brcb + Div/Mul 求 invScale/scale, 不取标量, 不调 SetDeqScale。
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_NORMAL_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_NORMAL_H_

#include "add_rms_norm_bias_dynamic_quant_ag_base.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormBiasDynamicQuantAGNormal : public KernelAddRmsNormBiasDynamicQuantAGBase {
public:
    __aicore__ inline KernelAddRmsNormBiasDynamicQuantAGNormal(TPipe* pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR bias,
        GM_ADDR yQuant, GM_ADDR scale, GM_ADDR x,
        GM_ADDR workspace, const AddRmsNormBiasDynamicQuantAGTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        SetTilingParams(tiling);
        this->blockIdx_ = GetBlockIdx();

        // Normal: block-factor 分配 (同 add_rms_norm_bias normal)
        if (this->blockIdx_ < (int32_t)(coreNum - 1)) {
            this->rowWork = blockFactor;
        } else if (this->blockIdx_ == (int32_t)(coreNum - 1)) {
            this->rowWork = latsBlockFactor;
        } else {
            this->rowWork = 0;
        }
        this->rowWork_ = this->rowWork;

        // 仅 AG 核: 跳过计算
        if (this->rowWork == 0) {
            this->InitAGParams(tiling);
            this->y1Out = yQuant;
            this->scale1Out = scale;
            this->InitUbBuffers();
            return;
        }

        uint32_t rowOffset = this->blockIdx_ * blockFactor;
        this->InitAGParams(tiling);
        this->y1Out = yQuant;
        this->scale1Out = scale;
        SetupGlobalBuffers(x1, x2, gamma, bias, x, rowOffset);
        this->InitUbBuffers();
    }

    // ---- 共享: tiling 参数解析 (Normal / SingleN 复用) ----
    __aicore__ inline void SetTilingParams(const AddRmsNormBiasDynamicQuantAGTilingData* tiling)
    {
        this->numCol = tiling->numCol;
        this->ubFactor = tiling->ubFactor;
        this->epsilon = tiling->epsilon;
        this->avgFactor = (numCol != 0) ? (1.0f / numCol) : 0.0f;
        this->blockFactor = tiling->blockFactor;
        this->latsBlockFactor = tiling->latsBlockFactor;
        this->coreNum = tiling->coreNum;
        this->hasX2 = tiling->hasX2;
        this->hasBias = tiling->hasBias;
        this->rowBatchSize = tiling->rowBatchSize;
        this->numColAlign = tiling->numColAlign;
        // int8 输出行步 (32 字节对齐 = 32 元素)
        this->outAlignLen = AlignUp<32u>(numCol);
    }

    // ---- 共享: 按行偏移设置 GM tensor (Normal / SingleN 复用) ----
    __aicore__ inline void SetupGlobalBuffers(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR bias,
                                              GM_ADDR x, uint32_t rowOffset)
    {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + rowOffset * numCol, rowWork * numCol);
        if (this->hasX2) {
            x2Gm.SetGlobalBuffer((__gm__ T*)x2 + rowOffset * numCol, rowWork * numCol);
        }
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if (this->hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T*)bias, numCol);
        }
        // 量化输出重定向到 HCCL 窗口 (与旧融合算子一致)
        uint32_t offsetScale = this->rowTotalNum * this->rowLen;
        yQuantGm.SetGlobalBuffer(
            (__gm__ int8_t*)this->buff[this->rankId] + rowOffset * numCol, rowWork * numCol);
        scaleGm.SetGlobalBuffer(
            (__gm__ float*)(this->buff[this->rankId] + offsetScale) + rowOffset, rowWork);
        // 中间输出 x (加法结果) -> 真实输出 GM
        xGm.SetGlobalBuffer((__gm__ T*)x + rowOffset * numCol, rowWork * numCol);
    }

    // 分配 4 个 UB 缓冲 (computeBuf + 3 个暂存), 总量由 host rowBatchSize 公式保证不溢出。
    __aicore__ inline void InitUbBuffers()
    {
        uint32_t ubF = this->ubFactor;
        uint32_t batch = this->rowBatchSize;
        // computeBuf: x1|x2 (各 ubF*T) + xFp32|sqx|tmp (各 ubF*float) + constArea(32 float)
        uint32_t computeBytes = (ubF * 4 + 32) * sizeof(float);  // = ubF*16 + 128
        pPipe->InitBuffer(computeBuf, computeBytes);
        if (this->rowWork_ == 0) { return; }  // AG-only 核无需暂存
        pPipe->InitBuffer(xStageBuf, batch * ubF * sizeof(T));
        pPipe->InitBuffer(yQuantStageBuf, batch * this->outAlignLen);           // int8
        pPipe->InitBuffer(scaleStageBuf, AlignUp<8u>(batch) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->rowWork_ == 0) {
            // 仅 AG 核
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

        // === AG 阶段: pipe reset + AllGather ===
        PipeBarrier<PIPE_ALL>();
        pPipe->Reset();
        pPipe->InitBuffer(this->copyBuf, USED_UB_SIZE);
        pPipe->InitBuffer(this->flagBuf, 32);
        this->ProcessAG();
    }

private:
    // ========================================================================
    // 常量缓冲初始化 (dynamic_quant DuplicateConst, 对称量化: 127 / 1-127)
    // ========================================================================
    __aicore__ inline void DuplicateConst(LocalTensor<float>& constScale, LocalTensor<float>& constInvScale)
    {
        Duplicate<float>(constScale, DYNAMIC_QUANT_INT8_SYM_SCALE, MAX_VALUE_NUM);        // 127.0
        PipeBarrier<PIPE_V>();
        Duplicate<float>(constInvScale, 1.0f / DYNAMIC_QUANT_INT8_SYM_SCALE, MAX_VALUE_NUM);  // 1/127
        PipeBarrier<PIPE_V>();
    }

    // ========================================================================
    // FP16 主流程
    // ========================================================================
    __aicore__ inline void ProcessFp16()
    {
        LocalTensor<float> ubLocal = computeBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
        LocalTensor<float> constScale = ubLocal[ubFactor * 4];
        LocalTensor<float> constInvScale = ubLocal[ubFactor * 4 + 8];
        LocalTensor<float> quantScaleTmp = ubLocal[ubFactor * 4 + 16];
        LocalTensor<float> brcbTmp = ubLocal[ubFactor * 4 + 24];

        LocalTensor<T> xStage = xStageBuf.Get<T>();
        LocalTensor<int8_t> yQuantStage = yQuantStageBuf.Get<int8_t>();
        LocalTensor<float> scaleStage = scaleStageBuf.Get<float>();

        DuplicateConst(constScale, constInvScale);

        uint32_t baseRow = 0;
        while (baseRow < rowWork) {
            uint32_t curBatch = (rowWork - baseRow < rowBatchSize) ? (rowWork - baseRow) : rowBatchSize;

            for (uint32_t b = 0; b < curBatch; b++) {
                uint32_t r = baseRow + b;
                LocalTensor<T> xSlot = xStage[b * ubFactor];  // 本行 x 暂存槽
                StageAddFp16(r, x1Local, x2Local, xFp32Local, sqxLocal, xSlot);
                StageRmsNormFp16(r, x1Local, x2Local, xFp32Local, sqxLocal, tmpLocal);
                StageDynamicQuantFp16(b, x1Local, xFp32Local, sqxLocal, tmpLocal,
                                      constScale, constInvScale, quantScaleTmp, brcbTmp,
                                      yQuantStage, scaleStage);
                PipeBarrier<PIPE_ALL>();
            }

            // 聚合写 GM (3 次批量 DataCopyPad, 禁止逐行写)
            FlushBatch(baseRow, curBatch, xStage, yQuantStage, scaleStage);
            baseRow += curBatch;
        }
    }

    // ---- Stage 1: Add (FP16) — 复刻 add_rms_norm_bias.h:125-130 ----
    // x = x1 + x2 (FP16 Add), 写入 xSlot (x 输出暂存); Cast fp32 和供 rmsnorm。
    __aicore__ inline void StageAddFp16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<T>& xSlot)
    {
        DataCopyCustom<T>(x1Local, x1Gm[row * numCol], numCol);
        event_t eventMTE2V_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_1);

        if (this->hasX2) {
            DataCopyCustom<T>(x2Local, x2Gm[row * numCol], numCol);
            event_t eventMTE2V_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_1);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_2);
            // FP16 Add 直接写入 x 暂存槽 (无 UB->UB 搬运)
            Add(xSlot, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            // Cast FP16 和 -> FP32 (供 rmsnorm, 与单算子 half 一致)
            Cast(xFp32Local, xSlot, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_1);
            // x = x1: 用 V pipe Cast 拷贝到暂存槽 (避免 UB->UB DataCopy)
            Cast(xSlot, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xSlot, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        }
    }

    // ---- Stage 2: RmsNorm (FP16) — 复刻 add_rms_norm_bias.h:269-318 Compute<half> ----
    // 输出 x1Local = FP16 y (= gamma*(x*rstd)+beta)。
    __aicore__ inline void StageRmsNormFp16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal)
    {
        // 载入 gamma 到 x2Local
        event_t eventVMTE2_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_1);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_1);
        DataCopyCustom<T>(x2Local, gammaGm, numCol);
        event_t eventMTE2V_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_1);

        // xFp32Local = fp32 和 (来自 StageAdd)
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE_F, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        // 取 rstd 标量
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        // x_norm = x * rstd (fp32)
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        // Cast FP32 -> FP16 y (单算子 half: Cast(yLocal, x_fp32, CAST_NONE))
        Cast(x1Local, xFp32Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // gamma 乘 (FP16, 单算子 half: Mul(yLocal, gammaLocal, yLocal))
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_1);
        Mul(x1Local, x1Local, x2Local, numCol);
        PipeBarrier<PIPE_V>();

        // bias 加 (FP16)
        if (this->hasBias) {
            event_t eventVMTE2_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2_2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2_2);
            DataCopyCustom<T>(x2Local, biasGm, numCol);
            event_t eventMTE2V_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_2);
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
        }
        // x1Local = FP16 y
    }

    // ---- Stage 3: DynamicQuant (FP16) — 复刻 dynamic_quant_db.h ComputeRowMax(half) + QuantizeRow ----
    // 输出: yQuantStage[b*outAlignLen], scaleStage[b]。
    __aicore__ inline void StageDynamicQuantFp16(uint32_t b,
        LocalTensor<T>& x1Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal,
        LocalTensor<float>& constScale, LocalTensor<float>& constInvScale,
        LocalTensor<float>& quantScaleTmp, LocalTensor<float>& brcbTmp,
        LocalTensor<int8_t>& yQuantStage, LocalTensor<float>& scaleStage)
    {
        // ---- ComputeRowMax (half 无 smooth 优化路径, dynamic_quant_db.h:289-304) ----
        // tempCast = Cast(inLocal, fp32)  -> xFp32Local
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        // half 域 Abs + ReduceMaxInplace
        Abs(x1Local, x1Local, numCol);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(x1Local, numCol);   // half 重载
        PipeBarrier<PIPE_V>();
        // temp = Cast(absmax, fp32) -> sqxLocal
        Cast(sqxLocal, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();

        // ---- QuantizeRow (dynamic_quant_db.h:309-330) ----
        QuantizeRowCore(b, xFp32Local, sqxLocal, tmpLocal,
                        constScale, constInvScale, quantScaleTmp, brcbTmp,
                        yQuantStage, scaleStage);
    }

    // ========================================================================
    // BF16 主流程
    // ========================================================================
    __aicore__ inline void ProcessBf16()
    {
        LocalTensor<float> ubLocal = computeBuf.Get<float>();
        LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
        LocalTensor<T> x1Local = xLocal[0];
        LocalTensor<T> x2Local = xLocal[ubFactor];
        LocalTensor<float> xFp32Local = ubLocal[ubFactor];
        LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
        LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
        LocalTensor<float> constScale = ubLocal[ubFactor * 4];
        LocalTensor<float> constInvScale = ubLocal[ubFactor * 4 + 8];
        LocalTensor<float> quantScaleTmp = ubLocal[ubFactor * 4 + 16];
        LocalTensor<float> brcbTmp = ubLocal[ubFactor * 4 + 24];

        LocalTensor<T> xStage = xStageBuf.Get<T>();
        LocalTensor<int8_t> yQuantStage = yQuantStageBuf.Get<int8_t>();
        LocalTensor<float> scaleStage = scaleStageBuf.Get<float>();

        DuplicateConst(constScale, constInvScale);

        uint32_t baseRow = 0;
        while (baseRow < rowWork) {
            uint32_t curBatch = (rowWork - baseRow < rowBatchSize) ? (rowWork - baseRow) : rowBatchSize;

            for (uint32_t b = 0; b < curBatch; b++) {
                uint32_t r = baseRow + b;
                LocalTensor<T> xSlot = xStage[b * ubFactor];
                StageAddBf16(r, x1Local, x2Local, xFp32Local, sqxLocal, xSlot);
                StageRmsNormBf16(r, x1Local, x2Local, xFp32Local, sqxLocal, tmpLocal);
                StageDynamicQuantBf16(b, x1Local, xFp32Local, sqxLocal, tmpLocal,
                                      constScale, constInvScale, quantScaleTmp, brcbTmp,
                                      yQuantStage, scaleStage);
                PipeBarrier<PIPE_ALL>();
            }

            FlushBatch(baseRow, curBatch, xStage, yQuantStage, scaleStage);
            baseRow += curBatch;
        }
    }

    // ---- Stage 1: Add (BF16) — 复刻 add_rms_norm_bias.h:131-145 ----
    // BF16: 两路 cast fp32 -> Add -> Cast RINT 回 bf16 写 xSlot; 保留 fp32 和供 rmsnorm (不 reload)。
    __aicore__ inline void StageAddBf16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<T>& xSlot)
    {
        DataCopyCustom<T>(x1Local, x1Gm[row * numCol], numCol);
        event_t eventMTE2V_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_1);

        if (this->hasX2) {
            DataCopyCustom<T>(x2Local, x2Gm[row * numCol], numCol);
            event_t eventMTE2V_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_2);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_1);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_2);
            // BF16: 两路 cast fp32, fp32 Add (单算子 CopyIn bf16)
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);   // fp32 和
            PipeBarrier<PIPE_V>();
            Cast(xSlot, xFp32Local, RoundMode::CAST_RINT, numCol);  // bf16 取整和 -> x 暂存
            PipeBarrier<PIPE_V>();
        } else {
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_1);
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xSlot, xFp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        }
        // xFp32Local 保留 fp32 和 (未取整) 供 rmsnorm —— 与单算子一致, 不做 bf16 reload
    }

    // ---- Stage 2: RmsNorm (BF16) — 复刻 add_rms_norm_bias.h:210-267 Compute<bfloat16_t> ----
    // gamma/beta 在 fp32 域乘加; 末尾 Cast RINT 出 bf16 y 到 x1Local。
    __aicore__ inline void StageRmsNormBf16(uint32_t row,
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal)
    {
        // 载入 gamma
        event_t eventVMTE2_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_1);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_1);
        DataCopyCustom<T>(x2Local, gammaGm, numCol);
        event_t eventMTE2V_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_2);

        // xFp32Local = fp32 和 (来自 StageAdd, 未取整)
        Mul(sqxLocal, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqxLocal, sqxLocal, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqxLocal, sqxLocal, tmpLocal, numCol);
        PipeBarrier<PIPE_V>();
        Adds(sqxLocal, sqxLocal, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqxLocal, sqxLocal, 1);
        Duplicate(tmpLocal, ONE_F, 1);
        PipeBarrier<PIPE_V>();
        Div(sqxLocal, tmpLocal, sqxLocal, 1);
        PipeBarrier<PIPE_V>();

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstdValue = sqxLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        // x_norm = x * rstd (fp32)
        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        // gamma -> fp32 (单算子 bf16: Cast(sqx, gammaLocal, CAST_NONE) 复用 sqx)
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_2);
        Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        // gamma 乘 (fp32)
        Mul(xFp32Local, xFp32Local, sqxLocal, numCol);
        PipeBarrier<PIPE_V>();

        // bias 加 (fp32)
        if (this->hasBias) {
            event_t eventVMTE2_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2_2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2_2);
            DataCopyCustom<T>(x2Local, biasGm, numCol);
            event_t eventMTE2V_3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_3);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_3);
            Cast(sqxLocal, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, sqxLocal, numCol);
            PipeBarrier<PIPE_V>();
        }

        // 末尾 Cast RINT 出 bf16 y -> x1Local (供 dynamic_quant, 与单算子 bf16 一致)
        Cast(x1Local, xFp32Local, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
    }

    // ---- Stage 3: DynamicQuant (BF16) — 复刻 dynamic_quant_db.h ComputeRowMax(bf16) + QuantizeRow ----
    __aicore__ inline void StageDynamicQuantBf16(uint32_t b,
        LocalTensor<T>& x1Local,
        LocalTensor<float>& xFp32Local, LocalTensor<float>& sqxLocal, LocalTensor<float>& tmpLocal,
        LocalTensor<float>& constScale, LocalTensor<float>& constInvScale,
        LocalTensor<float>& quantScaleTmp, LocalTensor<float>& brcbTmp,
        LocalTensor<int8_t>& yQuantStage, LocalTensor<float>& scaleStage)
    {
        // ---- ComputeRowMax (bf16, dynamic_quant_db.h:279-288) ----
        // tempCast = Cast(inLocal, fp32) -> xFp32Local
        Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
        PipeBarrier<PIPE_V>();
        // temp = Abs(tempCast) -> sqxLocal ; ReduceMaxInplace (float)
        Abs(sqxLocal, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(sqxLocal, numCol);   // float 重载
        PipeBarrier<PIPE_V>();

        // ---- QuantizeRow ----
        QuantizeRowCore(b, xFp32Local, sqxLocal, tmpLocal,
                        constScale, constInvScale, quantScaleTmp, brcbTmp,
                        yQuantStage, scaleStage);
    }

    // ========================================================================
    // QuantizeRow 公共实现 (dynamic_quant_db.h:309-330, 向量化, 不取标量, 不调 SetDeqScale)
    //   tempCast(xFp32Local) = fp32 输入; temp(sqxLocal) = 行 absmax。
    //   invScale = 127/absmax (Brcb+Div); y = round(tempCast*invScale); scale = absmax/127 (Mul)。
    // ========================================================================
    __aicore__ inline void QuantizeRowCore(uint32_t b,
        LocalTensor<float>& tempCast, LocalTensor<float>& temp, LocalTensor<float>& tmpLocal,
        LocalTensor<float>& constScale, LocalTensor<float>& constInvScale,
        LocalTensor<float>& quantScaleTmp, LocalTensor<float>& brcbTmp,
        LocalTensor<int8_t>& yQuantStage, LocalTensor<float>& scaleStage)
    {
        // 广播行 absmax 到 8 lane
        Brcb(brcbTmp, temp, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        // quantScaleTmp = 127 / absmax  (invScale)
        Div(quantScaleTmp, constScale, brcbTmp, MAX_VALUE_NUM);
        PipeBarrier<PIPE_V>();
        // tempCast = tempCast * invScale  (整行缩放, broadcast)
        Mul(tempCast, tempCast, quantScaleTmp, 64, (numCol + 63) >> 6, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        // FP32 -> INT16(round) -> half(round) -> INT8(trunc)
        LocalTensor<int32_t> tmpInt32 = tmpLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int16_t> tmpInt16 = tmpInt32.template ReinterpretCast<int16_t>();
        LocalTensor<half> tmpHalf = tmpLocal.template ReinterpretCast<half>();
        Cast(tmpInt16, tempCast, RoundMode::CAST_RINT, numCol);
        PipeBarrier<PIPE_V>();
        Cast(tmpHalf, tmpInt16, RoundMode::CAST_ROUND, numCol);
        PipeBarrier<PIPE_V>();
        // scale = absmax * (1/127)  (masked 写入 scaleStage 第 b 行, 与 db 一致)
        uint64_t mask[2] = { 1ULL << (b & 7), 0ULL };
        Mul(scaleStage[b & ~7u], brcbTmp, constInvScale, mask, 1, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        // yQuant -> 暂存 (行步 outAlignLen)
        Cast(yQuantStage[b * outAlignLen], tmpHalf, RoundMode::CAST_TRUNC, numCol);
        PipeBarrier<PIPE_V>();
    }

    // ========================================================================
    // 聚合写 GM: 一批 curBatch 行, 3 次批量 DataCopyPad (x / yQuant / scale)
    // (复刻 dynamic_quant_db.h CopyOutData 的多行批量写法, 禁止逐行写)
    // ========================================================================
    __aicore__ inline void FlushBatch(uint32_t baseRow, uint32_t curBatch,
        LocalTensor<T>& xStage, LocalTensor<int8_t>& yQuantStage, LocalTensor<float>& scaleStage)
    {
        // 确保 V 对暂存区的写入完成, 再 MTE3 读出写 GM
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);

        // x 暂存 -> 真实输出 GM (curBatch 行, 行步 numCol, T 元素)
        DataCopyExtParams xParams{static_cast<uint16_t>(curBatch), static_cast<uint16_t>(numCol * sizeof(T)), 0, 0, 0};
        DataCopyPad(xGm[baseRow * numCol], xStage, xParams);

        // yQuant 暂存 -> HCCL 窗口 (curBatch 行, 行步 numCol 字节, int8) —— 同 db CopyOutData
        DataCopyExtParams yqParams{static_cast<uint16_t>(curBatch), static_cast<uint16_t>(numCol), 0, 0, 0};
        DataCopyPad(yQuantGm[baseRow * numCol], yQuantStage, yqParams);

        // scale 暂存 -> HCCL 窗口 scale 段 (curBatch 个 float) —— 同 db CopyOutData
        DataCopyParams scaleParams{1, static_cast<uint16_t>(curBatch * sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[baseRow], scaleStage, scaleParams);

        // MTE3 读暂存完成后再允许下一批 V 写暂存
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
    }

protected:
    // 成员设为 protected, 供 SingleN 回退子类复用全部计算逻辑 (仅覆盖 Init 行分配)。
    TPipe* pPipe = nullptr;
    TBuf<TPosition::VECCALC> computeBuf;
    TBuf<TPosition::VECCALC> xStageBuf;
    TBuf<TPosition::VECCALC> yQuantStageBuf;
    TBuf<TPosition::VECCALC> scaleStageBuf;

    // GM tensors
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<int8_t> yQuantGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<T> xGm;       // add result

    // tiling 参数
    uint32_t numCol;
    uint32_t numColAlign;
    uint32_t ubFactor;
    uint32_t outAlignLen;
    float epsilon;
    float avgFactor;
    uint32_t blockFactor;
    uint32_t latsBlockFactor;
    uint32_t coreNum;
    uint32_t rowWork;
    uint32_t rowBatchSize;
};

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_NORMAL_H_
