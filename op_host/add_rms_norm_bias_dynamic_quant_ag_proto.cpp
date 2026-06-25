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
 * \file add_rms_norm_bias_dynamic_quant_ag_proto.cpp
 * \brief Shape and data type inference for AddRmsNormBiasDynamicQuantAG
 *
 * 3 outputs: y_quant(0), scale(1), x(2)
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_log.h"
#include "util/shape_util.h"
#include "error/ops_error.h"

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;

// AG attribute indices
static constexpr int ATTR_GROUP_SIZE = 3; // attr index 3 = groupSize

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4AddRmsNormBiasDynamicQuantAG(gert::InferShapeContext* context)
{
    OPS_LOG_D(context, "Begin to do InferShape4AddRmsNormBiasDynamicQuantAG");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    const gert::Shape* gammaShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    // get output shapes
    gert::Shape* yQuantShape = context->GetOutputShape(IDX_0);
    gert::Shape* scaleShape  = context->GetOutputShape(IDX_1);
    gert::Shape* xShape      = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, yQuantShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // x (add result): same shape as x1
    *xShape = *x1Shape;

    size_t xDimNum = x1Shape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();

    if (IsUnknownRank(*x1Shape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*yQuantShape);
        SetUnknownRank(*scaleShape);
        OPS_LOG_D(context, "End to do InferShape4AddRmsNormBiasDynamicQuantAG with unknown rank.");
        return GRAPH_SUCCESS;
    }

    OPS_CHECK(
        xDimNum < gammaDimNum,
        OPS_LOG_E(context->GetNodeName(),, "x dim num should not be smaller than gamma dim num."),
        return GRAPH_FAILED);

    // Get groupSize attribute for AG
    int64_t groupSize = 1;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pGroupSize = attrs->GetAttrPointer<int32_t>(ATTR_GROUP_SIZE);
        if (pGroupSize != nullptr && *pGroupSize > 0) {
            groupSize = *pGroupSize;
        }
    }

    // yQuant: same shape as x1, but first dim x groupSize (AllGather)
    *yQuantShape = *x1Shape;
    yQuantShape->SetDim(0, x1Shape->GetDim(0) * groupSize);

    // scale shape: x1 shape without last dimension, first dim x groupSize
    scaleShape->SetDimNum(xDimNum - 1);
    scaleShape->SetDim(0, x1Shape->GetDim(0) * groupSize);
    for (size_t i = 1; i < xDimNum - 1; i++) {
        scaleShape->SetDim(i, x1Shape->GetDim(i));
    }

    OPS_LOG_D(context->GetNodeName(), "End to do InferShape4AddRmsNormBiasDynamicQuantAG");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormBiasDynamicQuantAG(gert::InferDataTypeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataType4AddRmsNormBiasDynamicQuantAG");

    // yQuant (0): INT8
    context->SetOutputDataType(IDX_0, DT_INT8);

    // scale (1): FP32
    context->SetOutputDataType(IDX_1, DT_FLOAT);

    // x (2): same type as x1
    context->SetOutputDataType(IDX_2, context->GetInputDataType(IDX_0));

    OPS_LOG_D(context->GetNodeName(), "End to do InferDataType4AddRmsNormBiasDynamicQuantAG");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormBiasDynamicQuantAG)
    .InferShape(InferShape4AddRmsNormBiasDynamicQuantAG)
    .InferDataType(InferDataType4AddRmsNormBiasDynamicQuantAG);

} // namespace ops
