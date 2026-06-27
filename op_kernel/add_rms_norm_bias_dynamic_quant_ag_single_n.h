/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_single_n.h
 * \brief SingleN 回退 kernel for AddRmsNormBiasDynamicQuantAG (TilingKey 1301/3300)。
 *
 * 触发条件: 单行计算工作集或 db 单行放不下 UB, 或 numCol 非块对齐 (host primary=false)。
 * 行分配采用 head/tail (rowPerHeadCore / rowPerTailCore); rowBatchSize=1 (一批一行,
 * 仍走统一聚合写 GM 路径, 满足"不逐行写"语义)。
 *
 * 计算流程与 API 完全复用 Normal (继承), 仅覆盖 Init 行分配与行偏移。
 */

#ifndef ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_SINGLE_N_H_
#define ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_SINGLE_N_H_

#include "add_rms_norm_bias_dynamic_quant_ag_normal.h"

using namespace AscendC;

template <typename T>
class KernelAddRmsNormBiasDynamicQuantAGSingleN : public KernelAddRmsNormBiasDynamicQuantAGNormal<T> {
public:
    __aicore__ inline KernelAddRmsNormBiasDynamicQuantAGSingleN(TPipe* pipe)
        : KernelAddRmsNormBiasDynamicQuantAGNormal<T>(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR bias,
        GM_ADDR yQuant, GM_ADDR scale, GM_ADDR x,
        GM_ADDR workspace, const AddRmsNormBiasDynamicQuantAGTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->SetTilingParams(tiling);
        // SingleN 专用分配参数
        this->headCoreNum = tiling->headCoreNum;
        this->rowPerHeadCore = tiling->rowPerHeadCore;
        this->rowPerTailCore = tiling->rowPerTailCore;
        this->blockIdx_ = GetBlockIdx();

        // head/tail 分配: 前 headCoreNum 个核处理 rowPerHeadCore 行, 其余处理 rowPerTailCore 行
        uint32_t rowOffset = 0;
        if (this->headCoreNum == 0) {
            // 完美整除: 所有核同 rowPerHeadCore
            this->rowWork = rowPerHeadCore;
            rowOffset = this->blockIdx_ * rowPerHeadCore;
        } else if (this->blockIdx_ < (int32_t)this->headCoreNum) {
            this->rowWork = rowPerHeadCore;
            rowOffset = this->blockIdx_ * rowPerHeadCore;
        } else if (this->blockIdx_ < (int32_t)this->coreNum) {
            this->rowWork = rowPerTailCore;
            rowOffset = this->headCoreNum * rowPerHeadCore +
                        (this->blockIdx_ - this->headCoreNum) * rowPerTailCore;
        } else {
            this->rowWork = 0;  // 仅 AG 核
        }
        this->rowWork_ = this->rowWork;

        // 仅 AG 核
        if (this->rowWork == 0) {
            this->InitAGParams(tiling);
            this->y1Out = yQuant;
            this->scale1Out = scale;
            this->InitUbBuffers();
            return;
        }

        this->InitAGParams(tiling);
        this->y1Out = yQuant;
        this->scale1Out = scale;
        this->SetupGlobalBuffers(x1, x2, gamma, bias, x, rowOffset);
        this->InitUbBuffers();
    }

private:
    uint32_t headCoreNum;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
};

#endif // ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_SINGLE_N_H_
