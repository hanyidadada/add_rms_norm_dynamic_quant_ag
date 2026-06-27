/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_def.cpp
 * \brief Operator definition for AddRmsNormBiasDynamicQuantAG
 *
 * Fused operator: Add(x1+x2) + RMS Norm + Dynamic Quant + AllGather
 *  - TilingKey 组合编码: rmsKey*100 + quantKey
 *      (rmsKey 复用 add_rms_norm_bias 的 dtypeKey*10+modeKey;
 *       quantKey 复用 dynamic_quant 的 db 对称量化 key)
 *  - UB 内聚合多行计算, 批量聚合 DataCopy 写 GM
 * Targets: ascend910b, ascend910_93
 * Dtypes: FP16, BF16
 * I/O 契约与旧融合算子 AddRmsNormBiasDynamicQuantAG 完全一致。
 */

#include "register/op_def_registry.h"

namespace ops {
class AddRmsNormBiasDynamicQuantAG : public OpDef {
public:
    explicit AddRmsNormBiasDynamicQuantAG(const char* name) : OpDef(name)
    {
        // Input: x1 - first Add input
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Input: x2 - second Add input (optional)
        this->Input("x2")
            .ParamType(OPTIONAL)
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

        // Input: bias - RmsNorm bias (optional)
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Output: y_quant - quantized result (INT8), first dim x groupSize via AllGather
        this->Output("y_quant")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // Output: scale - quantization scale (FP32), first dim x groupSize via AllGather
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // Output: x - Add result (same type as x1, shape unchanged)
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Attributes (order identical to old fused op)
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("groupSize").AttrType(OPTIONAL).Int(0);

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

        this->MC2().HcclGroup("group");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(AddRmsNormBiasDynamicQuantAG);
} // namespace ops
