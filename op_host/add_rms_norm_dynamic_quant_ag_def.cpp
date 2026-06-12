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
 * \file add_rms_norm_dynamic_quant_ag_def.cpp
 * \brief Operator definition for AddRmsNormDynamicQuantAG
 */

#include "register/op_def_registry.h"

namespace ops {
class AddRmsNormDynamicQuantAG : public OpDef {
public:
    explicit AddRmsNormDynamicQuantAG(const char* name) : OpDef(name)
    {
        // Input: x1 - first Add input
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Input: x2 - second Add input
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Input: gamma - RmsNorm scaling factor
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Output: y_quant - quantized result (INT8), first dim × groupSize via AllGather
        this->Output("y_quant")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // Output: scale - quantization scale (FP32), first dim × groupSize via AllGather
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // Output: y_add - Add result (same type as x1, shape unchanged)
        this->Output("y_add")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Output: rstd - RmsNorm reciprocal standard deviation (FP32, shape unchanged)
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Attributes
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);

        // === AG 相关属性 ===
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("groupSize").AttrType(OPTIONAL).Int(0);

        // Platform config with dynamic shape/format support
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        // === MC2 通信配置 ===
        this->MC2().HcclGroup("group");
         // Platform: Ascend 910B
        this->AICore().AddConfig("ascend910b", aicore_config);

        // Platform: Ascend 910_93
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(AddRmsNormDynamicQuantAG);
} // namespace ops
