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
 * \file add_rms_norm_bias_dynamic_quant_ag_tiling.cpp
 * \brief Tiling strategy for AddRmsNormBiasDynamicQuantAG fusion operator
 *
 * Tiling key encoding: (dtype_key * 10 + mode_key)
 *   dtype_key: 1 = half, 3 = bf16
 *   mode_key:  0 = SingleN, 1 = MultiN
 * Valid keys: 10, 30, 11, 31
 */
#include "add_rms_norm_bias_dynamic_quant_ag_info.h"
#include "add_rms_norm_bias_dynamic_quant_ag_tiling.h"

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_log.h"

using namespace ge;

namespace optiling {

// Tiling key encoding
constexpr uint32_t DTYPE_KEY_HALF  = 1;
constexpr uint32_t DTYPE_KEY_BF16  = 3;
constexpr uint32_t MODE_SINGLE_N   = 0;
constexpr uint32_t MODE_MULTI_N    = 1;

constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t UB_RESERVED = 1024;                  // reserved UB space
constexpr uint32_t SYS_WORKSPACE = 16 * 1024 * 1024;    // 16MB system workspace
constexpr uint32_t USR_WORKSPACE = 256;

// UB size per row estimate coefficients
// SingleN layout: x1Local(2B) + x2Local(2B) + xFp32Local(4B) + sqxLocal(4B) + tmpLocal(4B) + outInt8Local(1B)
// = ubFactor * (2+2+4+4+4+1) = ubFactor * 17 bytes
constexpr uint32_t UB_PER_ROW_FP16_COEFF = 17;
constexpr uint32_t UB_PER_ROW_BF16_COEFF = 17;

// I/O indices
static constexpr int IDX_X1      = 0;
static constexpr int IDX_X2      = 1;
static constexpr int IDX_GAMMA   = 2;
static constexpr int IDX_YQUANT  = 0;
static constexpr int IDX_SCALE   = 1;
static constexpr int IDX_X       = 2;   // add result
static constexpr int IDX_Y       = 3;   // rmsnorm result
static constexpr int IDX_RSTD    = 4;

// AG attribute indices
static constexpr int GROUP_IDX = 2;
static constexpr int GROUP_SIZE_IDX = 3;

// ========== Utility Functions ==========

template <uint32_t base, typename T = uint32_t>
static T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

static uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

// ========== Parameter Validation ==========

static bool CheckNullptr(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape     = context->GetInputShape(IDX_X1);
    const gert::StorageShape* x2Shape     = context->GetInputShape(IDX_X2);
    const gert::StorageShape* gammaShape  = context->GetInputShape(IDX_GAMMA);
    const gert::StorageShape* yQuantShape = context->GetOutputShape(IDX_YQUANT);
    const gert::StorageShape* scaleShape  = context->GetOutputShape(IDX_SCALE);
    const gert::StorageShape* xShape      = context->GetOutputShape(IDX_X);
    const gert::StorageShape* yShape      = context->GetOutputShape(IDX_Y);
    const gert::StorageShape* rstdShape   = context->GetOutputShape(IDX_RSTD);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yQuantShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    return true;
}

static bool CheckDataType(gert::TilingContext* context)
{
    auto x1Dtype    = context->GetInputDesc(IDX_X1)->GetDataType();
    auto x2Dtype    = context->GetInputDesc(IDX_X2)->GetDataType();
    auto gammaDtype = context->GetInputDesc(IDX_GAMMA)->GetDataType();

    OP_CHECK_IF(
        x1Dtype != x2Dtype,
        OPS_LOG_E(context->GetNodeName(), "x1 and x2 must have the same data type."),
        return false);

    OP_CHECK_IF(
        x1Dtype != gammaDtype,
        OPS_LOG_E(context->GetNodeName(), "x1 and gamma must have the same data type."),
        return false);

    OP_CHECK_IF(
        x1Dtype != DT_FLOAT16 && x1Dtype != DT_BF16,
        OPS_LOG_E(context->GetNodeName(), "data type must be FP16 or BF16."),
        return false);

    return true;
}

static bool CheckInputOutputDim(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape     = context->GetInputShape(IDX_X1);
    const gert::StorageShape* x2Shape     = context->GetInputShape(IDX_X2);
    const gert::StorageShape* gammaShape  = context->GetInputShape(IDX_GAMMA);
    const gert::StorageShape* yQuantShape = context->GetOutputShape(IDX_YQUANT);
    const gert::StorageShape* scaleShape  = context->GetOutputShape(IDX_SCALE);
    const gert::StorageShape* xShape      = context->GetOutputShape(IDX_X);
    const gert::StorageShape* yShape      = context->GetOutputShape(IDX_Y);
    const gert::StorageShape* rstdShape   = context->GetOutputShape(IDX_RSTD);

    size_t x1DimNum     = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum     = x2Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum  = gammaShape->GetStorageShape().GetDimNum();
    size_t yQuantDimNum = yQuantShape->GetStorageShape().GetDimNum();
    size_t scaleDimNum  = scaleShape->GetStorageShape().GetDimNum();
    size_t xDimNum      = xShape->GetStorageShape().GetDimNum();
    size_t yDimNum      = yShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum   = rstdShape->GetStorageShape().GetDimNum();

    // x1 dims should be 2-8
    OP_CHECK_IF(
        x1DimNum < 2 || x1DimNum > 8,
        OPS_LOG_E(context->GetNodeName(), "x1 dim num must be in range [2, 8]."),
        return false);

    // x1, x2, yQuant, x, y must have same dims
    OP_CHECK_IF(
        x1DimNum != x2DimNum || x1DimNum != yQuantDimNum || x1DimNum != xDimNum || x1DimNum != yDimNum,
        OPS_LOG_E(context->GetNodeName(), "x1, x2, yQuant, x, y must have same dims."),
        return false);

    // gamma dims: must be <= x1 dims
    OP_CHECK_IF(
        gammaDimNum > x1DimNum,
        OPS_LOG_E(context->GetNodeName(), "gamma dim num should not be greater than x1 dim num."),
        return false);

    // scale dims = x1 dims - 1
    OP_CHECK_IF(
        scaleDimNum != x1DimNum - 1,
        OPS_LOG_E(context->GetNodeName(), "scale dim num should be x1 dim num - 1."),
        return false);

    // rstd dims = x1 dims
    OP_CHECK_IF(
        rstdDimNum != x1DimNum,
        OPS_LOG_E(context->GetNodeName(), "rstd dim num should be same as x1 dim num."),
        return false);

    // last dim of x1 and gamma must match
    OP_CHECK_IF(
        x1Shape->GetStorageShape().GetDim(x1DimNum - 1) !=
            gammaShape->GetStorageShape().GetDim(gammaDimNum - 1),
        OPS_LOG_E(context->GetNodeName(), "Last dim of x1 and gamma must be the same."),
        return false);

    return true;
}

// ========== Parameter Extraction ==========

static void GetCompileParameters(gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize, uint32_t& sysWorkspaceSize)
{
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormBiasDynamicQuantAGCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        numCore = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize  = ptrCompileInfo->maxUbSize;
    }
    ubSize -= UB_RESERVED;
}

static void CalculateRowAndColParams(gert::TilingContext* context, uint32_t& numRow, uint32_t& numCol)
{
    const gert::Shape x1Shape = context->GetInputShape(IDX_X1)->GetStorageShape();
    const gert::Shape gammaShape = context->GetInputShape(IDX_GAMMA)->GetStorageShape();

    numCol = gammaShape.GetShapeSize();

    size_t x1DimNum = x1Shape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    numRow = 1U;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; ++i) {
        numRow *= x1Shape.GetDim(i);
    }
}

static float GetEpsilon(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 1e-6f;
    }
    float epsilon = *attrs->GetFloat(0);
    return (epsilon >= 0) ? epsilon : 1e-6f;
}

static uint32_t GetDstType(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return static_cast<uint32_t>(DT_INT8);
    }
    const int32_t* pDstType = attrs->GetAttrPointer<int32_t>(1);
    if (pDstType == nullptr) {
        return static_cast<uint32_t>(DT_INT8);
    }
    return static_cast<uint32_t>(*pDstType);
}

static uint32_t GetDtypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case DT_FLOAT16: return DTYPE_KEY_HALF;
        case DT_BF16:    return DTYPE_KEY_BF16;
        default:         return DTYPE_KEY_HALF;
    }
}

// ========== Tiling Strategy ==========

static void CalculateMultiCoreDistribution(
    uint32_t numRow, uint32_t numCore,
    uint32_t& headCoreNum, uint32_t& rowPerHeadCore, uint32_t& rowPerTailCore,
    uint32_t& useCoreNum)
{
    // When numRow < numCore, limit cores to numRow (each core gets at least 1 row)
    uint32_t effectiveNumCore = std::min(numRow, numCore);
    useCoreNum = effectiveNumCore;
    rowPerHeadCore = CeilDiv(numRow, effectiveNumCore);
    uint32_t tailCoreNum = rowPerHeadCore * effectiveNumCore - numRow;
    if (tailCoreNum == 0) {
        headCoreNum = effectiveNumCore;
        rowPerTailCore = rowPerHeadCore;
    } else {
        headCoreNum = effectiveNumCore - 1;
        rowPerTailCore = numRow - rowPerHeadCore * (effectiveNumCore - 1);
    }
}

static uint32_t DetermineModeAndRows(
    uint32_t numCol, uint64_t ubSize, ge::DataType dataType,
    uint32_t& multiRowNum, uint32_t& ubFactor)
{
    uint32_t coeff = (dataType == DT_BF16) ? UB_PER_ROW_BF16_COEFF : UB_PER_ROW_FP16_COEFF;

    // Align numCol to block size
    ubFactor = AlignUp<BLOCK_ALIGN_NUM>(numCol);

    // Estimate UB required per row
    uint64_t ubPerRow = static_cast<uint64_t>(ubFactor) * coeff;

    // Calculate max rows that fit in UB
    uint32_t maxRows = static_cast<uint32_t>(ubSize / ubPerRow);

    if (maxRows < 1) {
        multiRowNum = 1;
        ubFactor = AlignUp<BLOCK_ALIGN_NUM>(numCol);
        return MODE_SINGLE_N;
    } else if (maxRows == 1) {
        multiRowNum = 1;
        return MODE_SINGLE_N;
    } else {
        multiRowNum = maxRows;
        return MODE_MULTI_N;
    }
}

// ========== Tiling Prepare ==========

static ge::graphStatus TilingPrepareAddRmsNormBiasDynamicQuantAG(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(nullptr == context, OPS_LOG_E(context->GetNodeName(), "Context is null"), return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "Enter TilingPrepareAddRmsNormBiasDynamicQuantAG.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfoPtr == nullptr, OPS_LOG_E(context->GetNodeName(), "PlatformInfoPtr is null"),
               return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<AddRmsNormBiasDynamicQuantAGCompileInfo>();
    OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context->GetNodeName(), "CompileInfoPtr is null"),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

// ========== Main Tiling Entry ==========

static ge::graphStatus TilingAddRmsNormBiasDynamicQuantAG(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Enter TilingAddRmsNormBiasDynamicQuantAG");

    // 1. Parameter validation
    OP_CHECK_IF(!CheckNullptr(context), OPS_LOG_E(context->GetNodeName(), "Input shape invalid (nullptr)."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDataType(context), OPS_LOG_E(context->GetNodeName(), "Data type check failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputOutputDim(context), OPS_LOG_E(context->GetNodeName(), "Dimension check failed."), return ge::GRAPH_FAILED);

    // 2. Get compilation parameters
    uint32_t numCore = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSize = 0;
    GetCompileParameters(context, numCore, ubSize, sysWorkspaceSize);

    // 3. Extract shape parameters
    uint32_t numRow = 0;
    uint32_t numCol = 0;
    CalculateRowAndColParams(context, numRow, numCol);

    // 4. Extract attributes
    float epsilon = GetEpsilon(context);
    uint32_t dstType = GetDstType(context);

    // 5. Get data type
    auto dataType = context->GetInputDesc(IDX_X1)->GetDataType();
    uint32_t dtypeKey = GetDtypeKey(dataType);

    // 6. Calculate multi-core distribution
    uint32_t headCoreNum = 0;
    uint32_t rowPerHeadCore = 0;
    uint32_t rowPerTailCore = 0;
    uint32_t useCoreNum = 0;
    CalculateMultiCoreDistribution(numRow, numCore, headCoreNum, rowPerHeadCore, rowPerTailCore, useCoreNum);

    // 7. Determine mode and UB parameters
    uint32_t multiRowNum = 1;
    uint32_t ubFactor = 0;
    uint32_t modeKey = DetermineModeAndRows(numCol, ubSize, dataType, multiRowNum, ubFactor);

    // 8. Calculate tiling key
    uint32_t tilingKey = (dtypeKey * 10) + modeKey;
    context->SetTilingKey(tilingKey);

    // 9. Extract AG attributes and compute MC2 parameters
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(GROUP_IDX);
    auto groupSizePtr = attrs->GetAttrPointer<int>(GROUP_SIZE_IDX);

    const gert::StorageShape* x1Shape = context->GetInputShape(IDX_X1);
    uint64_t rowLen = x1Shape->GetStorageShape().GetDim(x1Shape->GetStorageShape().GetDimNum() - 1);
    size_t dimNum = x1Shape->GetStorageShape().GetDimNum() - 1;
    uint64_t rowTotalNum = 1;
    for (size_t i = 0; i < dimNum; i++) {
        rowTotalNum *= x1Shape->GetStorageShape().GetDim(i);
    }

    // 10. Build tiling data struct
    AddRmsNormBiasDynamicQuantAGTilingData *tilingData = context->GetTilingData<AddRmsNormBiasDynamicQuantAGTilingData>();

    tilingData->groupSize      = *groupSizePtr;
    tilingData->rowLen         = rowLen;
    tilingData->rowTotalNum    = rowTotalNum;
    tilingData->numRow         = numRow;
    tilingData->numCol         = numCol;
    tilingData->epsilon        = epsilon;
    tilingData->avgFactor      = (numCol == 0) ? 0.0f : (1.0f / static_cast<float>(numCol));
    tilingData->dstType        = dstType;
    tilingData->coreNum        = useCoreNum;
    tilingData->headCoreNum    = headCoreNum;
    tilingData->rowPerHeadCore = rowPerHeadCore;
    tilingData->rowPerTailCore = rowPerTailCore;
    tilingData->multiRowNum    = multiRowNum;
    tilingData->ubFactor       = ubFactor;

    // 11. MC2 AlltoAll communication configuration
    uint32_t opType = 8; // batch write
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();

    std::string algConfig;
    if (aivNum < 24) {
        algConfig = "AlltoAll=level0:fullmesh";
    } else {
        algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    }
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    // 12. Set block dim (must be >= groupSize for AG)
    uint32_t usedcore = std::max(useCoreNum, (uint32_t)*groupSizePtr);
    context->SetBlockDim(usedcore);

    // 13. Set workspace size
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = USR_WORKSPACE + sysWorkspaceSize;

    // 14. Log results
    OPS_LOG_I(context->GetNodeName(), "Tiling Key: %u", tilingKey);
    OPS_LOG_I(context->GetNodeName(), "Block Dim: %u (useCore: %u, groupSize: %u)", usedcore, useCoreNum, *groupSizePtr);
    OPS_LOG_I(context->GetNodeName(), "numRow: %u, numCol: %u, multiRowNum: %u, ubFactor: %u",
            numRow, numCol, multiRowNum, ubFactor);
    OPS_LOG_I(context->GetNodeName(), "rowLen: %llu, rowTotalNum: %llu, groupSize: %llu",
            rowLen, rowTotalNum, (uint64_t)*groupSizePtr);
    OPS_LOG_I(context->GetNodeName(), "epsilon: %f, avgFactor: %f", epsilon, tilingData->avgFactor);
    OPS_LOG_I(context->GetNodeName(), "headCoreNum: %u, rowPerHeadCore: %u, rowPerTailCore: %u",
            headCoreNum, rowPerHeadCore, rowPerTailCore);
    OPS_LOG_I(context->GetNodeName(), "Exit TilingAddRmsNormBiasDynamicQuantAG");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormBiasDynamicQuantAG)
    .Tiling(TilingAddRmsNormBiasDynamicQuantAG)
    .TilingParse<AddRmsNormBiasDynamicQuantAGCompileInfo>(TilingPrepareAddRmsNormBiasDynamicQuantAG);

} // namespace optiling
