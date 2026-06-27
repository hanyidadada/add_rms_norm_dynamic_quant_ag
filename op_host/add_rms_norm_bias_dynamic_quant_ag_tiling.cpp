/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_tiling.cpp
 * \brief Tiling strategy for AddRmsNormBiasDynamicQuantAG
 *
 * ============ TilingKey 组合编码 (复用两个单算子分支判定) ============
 *   TilingKey = rmsKey * 100 + quantKey
 *
 *   rmsKey   <- add_rms_norm_bias 的 dtypeKey*10 + modeKey
 *                dtypeKey: 1=half, 3=bf16 ; modeKey: 0=normal, 3=single_n
 *                normal  -> 10(half) / 30(bf16)
 *                single_n-> 13(half) / 33(bf16)
 *   quantKey <- dynamic_quant 的 SetTilingKey 输出
 *                useDb=true  -> 3(half) / 2(bf16)   (db 双缓冲对称量化)
 *                useDb=false -> 1(half) / 0(bf16)   (非 db, 超大行回退)
 *
 *   主分支 (Normal+db):   rowFitsCompute && useDb && numColBlockAligned
 *       half -> rmsKey=10, quantKey=3 -> 1003
 *       bf16 -> rmsKey=30, quantKey=2 -> 3002
 *   回退分支 (SingleN):   否则
 *       half -> rmsKey=13, quantKey=1 -> 1301
 *       bf16 -> rmsKey=33, quantKey=0 -> 3300
 *
 * ============ rowBatchSize 动态计算 (UB 剩余空间) ============
 *   单行计算工作集 computeWorkspace = ubFactor*16 + CONST_OVERHEAD (每行复用)
 *   每行暂存 stagingPerRow = ubFactor*sizeof(T) [x] + outAlignLen [yQuant] + sizeof(float) [scale]
 *   remainUB   = ubSize - computeWorkspace
 *   rowBatchSize = clamp(remainUB / stagingPerRow, 1, min(blockFactor, 255))
 *   SingleN 回退: rowBatchSize = 1
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

// ---------- dtype / mode keys (与单算子一致) ----------
constexpr uint32_t DTYPE_KEY_HALF = 1;
constexpr uint32_t DTYPE_KEY_BF16 = 3;

// rms mode (add_rms_norm_bias)
constexpr uint32_t MODE_NORMAL   = 0;
constexpr uint32_t MODE_SINGLE_N = 3;

// quant keys (dynamic_quant SetTilingKey)
constexpr uint32_t QUANT_DB_HALF     = 3;  // useDb=true,  half
constexpr uint32_t QUANT_DB_BF16     = 2;  // useDb=true,  bf16
constexpr uint32_t QUANT_NONDB_HALF  = 1;  // useDb=false, half
constexpr uint32_t QUANT_NONDB_BF16  = 0;  // useDb=false, bf16

// ---------- 对齐与 UB 常量 ----------
constexpr uint32_t BLOCK_ALIGN_NUM  = 16;     // half/bf16 块对齐 (元素)
constexpr uint32_t INT8_ALIGN_NUM   = 32;     // int8 块对齐 (元素, 32 字节)
constexpr uint32_t UB_RESERVED      = 1024;   // 预留 UB
constexpr uint32_t SYS_WORKSPACE    = 16 * 1024 * 1024;
constexpr uint32_t USR_WORKSPACE    = 256;
constexpr uint32_t MAX_ROW_BATCH    = 255;    // reduce/broadcast 指令最大重复数
constexpr uint32_t CONST_OVERHEAD   = 1024;   // 单行常量/reduce 辅助缓冲开销 (字节)

// 单行计算工作集系数: x1(T)+x2(T)+xFp32(f)+sqx(f)+tmp(f) = (2+2+4+4+4)=16 B/elem
constexpr uint32_t COMPUTE_COEFF = 16;
// dynamic_quant db 单行 UB 系数 (无 smooth, FP16_DB_UB_SIZE=13, dynamic_quant_tiling.cpp:43-53)
constexpr uint32_t DB_UB_COEFF = 13;

// ---------- I/O 索引 ----------
static constexpr int IDX_X1     = 0;
static constexpr int IDX_X2     = 1;
static constexpr int IDX_GAMMA  = 2;
static constexpr int IDX_BIAS   = 3;
static constexpr int IDX_YQUANT = 0;
static constexpr int IDX_SCALE  = 1;
static constexpr int IDX_X      = 2;

static constexpr int EPSILON_IDX     = 0;
static constexpr int GROUP_IDX       = 1;
static constexpr int GROUP_SIZE_IDX  = 2;

// ========== 工具函数 ==========
template <uint32_t base, typename T = uint32_t>
static T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

static uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

// ========== 参数校验 (与旧融合算子一致) ==========
static bool CheckNullptr(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape     = context->GetInputShape(IDX_X1);
    const gert::StorageShape* gammaShape  = context->GetInputShape(IDX_GAMMA);
    const gert::StorageShape* yQuantShape = context->GetOutputShape(IDX_YQUANT);
    const gert::StorageShape* scaleShape  = context->GetOutputShape(IDX_SCALE);
    const gert::StorageShape* xShape      = context->GetOutputShape(IDX_X);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yQuantShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    return true;
}

static bool CheckDataType(gert::TilingContext* context)
{
    auto x1Dtype    = context->GetInputDesc(IDX_X1)->GetDataType();
    auto gammaDtype = context->GetInputDesc(IDX_GAMMA)->GetDataType();

    OP_CHECK_IF(
        x1Dtype != gammaDtype,
        OPS_LOG_E(context->GetNodeName(), "x1 and gamma must have the same data type."),
        return false);
    OP_CHECK_IF(
        x1Dtype != DT_FLOAT16 && x1Dtype != DT_BF16,
        OPS_LOG_E(context->GetNodeName(), "data type must be FP16 or BF16."),
        return false);

    auto x2Desc = context->GetOptionalInputDesc(IDX_X2);
    if (x2Desc != nullptr) {
        OP_CHECK_IF(
            x1Dtype != x2Desc->GetDataType(),
            OPS_LOG_E(context->GetNodeName(), "x1 and x2 must have the same data type."),
            return false);
    }
    auto biasDesc = context->GetOptionalInputDesc(IDX_BIAS);
    if (biasDesc != nullptr) {
        OP_CHECK_IF(
            x1Dtype != biasDesc->GetDataType(),
            OPS_LOG_E(context->GetNodeName(), "x1 and bias must have the same data type."),
            return false);
    }
    return true;
}

static bool CheckInputOutputDim(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape     = context->GetInputShape(IDX_X1);
    const gert::StorageShape* gammaShape  = context->GetInputShape(IDX_GAMMA);
    const gert::StorageShape* yQuantShape = context->GetOutputShape(IDX_YQUANT);
    const gert::StorageShape* scaleShape  = context->GetOutputShape(IDX_SCALE);
    const gert::StorageShape* xShape      = context->GetOutputShape(IDX_X);

    size_t x1DimNum     = x1Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum  = gammaShape->GetStorageShape().GetDimNum();
    size_t yQuantDimNum = yQuantShape->GetStorageShape().GetDimNum();
    size_t scaleDimNum  = scaleShape->GetStorageShape().GetDimNum();
    size_t xDimNum      = xShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(
        x1DimNum < 2 || x1DimNum > 8,
        OPS_LOG_E(context->GetNodeName(), "x1 dim num must be in range [2, 8]."),
        return false);
    OP_CHECK_IF(
        x1DimNum != yQuantDimNum || x1DimNum != xDimNum,
        OPS_LOG_E(context->GetNodeName(), "x1, yQuant, x must have same dims."),
        return false);
    OP_CHECK_IF(
        gammaDimNum > x1DimNum,
        OPS_LOG_E(context->GetNodeName(), "gamma dim num should not be greater than x1 dim num."),
        return false);
    OP_CHECK_IF(
        scaleDimNum != x1DimNum - 1,
        OPS_LOG_E(context->GetNodeName(), "scale dim num should be x1 dim num - 1."),
        return false);
    OP_CHECK_IF(
        x1Shape->GetStorageShape().GetDim(x1DimNum - 1) !=
            gammaShape->GetStorageShape().GetDim(gammaDimNum - 1),
        OPS_LOG_E(context->GetNodeName(), "Last dim of x1 and gamma must be the same."),
        return false);
    return true;
}

// ========== 参数提取 ==========
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
    float epsilon = *attrs->GetFloat(EPSILON_IDX);
    return (epsilon >= 0) ? epsilon : 1e-6f;
}

static uint32_t GetDtypeKey(ge::DataType dataType)
{
    switch (dataType) {
        case DT_FLOAT16: return DTYPE_KEY_HALF;
        case DT_BF16:    return DTYPE_KEY_BF16;
        default:         return DTYPE_KEY_HALF;
    }
}

static uint32_t GetQuantDbKey(ge::DataType dataType)
{
    // dynamic_quant useDb=true 分支: half->3, bf16->2
    return (dataType == DT_BF16) ? QUANT_DB_BF16 : QUANT_DB_HALF;
}

static uint32_t GetQuantNonDbKey(ge::DataType dataType)
{
    // dynamic_quant useDb=false 分支: half->1, bf16->0
    return (dataType == DT_BF16) ? QUANT_NONDB_BF16 : QUANT_NONDB_HALF;
}

// ========== 核心分配 (SingleN head/tail) ==========
static void CalculateMultiCoreDistribution(
    uint32_t numRow, uint32_t numCore,
    uint32_t& headCoreNum, uint32_t& rowPerHeadCore, uint32_t& rowPerTailCore,
    uint32_t& useCoreNum)
{
    uint32_t effectiveNumCore = std::min(numRow, numCore);
    useCoreNum = effectiveNumCore;
    rowPerHeadCore = CeilDiv(numRow, effectiveNumCore);
    uint32_t remainder = numRow % effectiveNumCore;
    if (remainder == 0) {
        headCoreNum = 0;
        rowPerTailCore = rowPerHeadCore;
    } else {
        headCoreNum = remainder;
        rowPerTailCore = rowPerHeadCore - 1;
    }
}

// ========== TilingParse ==========
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

// ========== 主 Tiling 入口 ==========
static ge::graphStatus TilingAddRmsNormBiasDynamicQuantAG(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Enter TilingAddRmsNormBiasDynamicQuantAG");

    // 1. 校验
    OP_CHECK_IF(!CheckNullptr(context), OPS_LOG_E(context->GetNodeName(), "Input shape invalid (nullptr)."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckDataType(context), OPS_LOG_E(context->GetNodeName(), "Data type check failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputOutputDim(context), OPS_LOG_E(context->GetNodeName(), "Dimension check failed."), return ge::GRAPH_FAILED);

    // 2. 编译参数
    uint32_t numCore = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSize = 0;
    GetCompileParameters(context, numCore, ubSize, sysWorkspaceSize);

    // 3. 形状参数
    uint32_t numRow = 0;
    uint32_t numCol = 0;
    CalculateRowAndColParams(context, numRow, numCol);

    // 4. 属性
    float epsilon = GetEpsilon(context);

    // 5. 数据类型
    auto dataType = context->GetInputDesc(IDX_X1)->GetDataType();
    uint32_t dtypeKey = GetDtypeKey(dataType);
    uint32_t sizeofT = (dataType == DT_BF16) ? 2u : 2u;  // half/bf16 均 2 字节

    // 6. 对齐列数
    uint32_t numColAlign = AlignUp<BLOCK_ALIGN_NUM>(numCol);       // half/bf16 行步 (元素)
    uint32_t outAlignLen = AlignUp<INT8_ALIGN_NUM>(numCol);        // int8 行步 (元素)
    uint32_t ubFactor = numColAlign;

    // 7. 复用单算子分支判定
    //    (a) rms normal 判定: 单行计算工作集能放进 UB
    uint64_t computeWorkspace = static_cast<uint64_t>(ubFactor) * COMPUTE_COEFF + CONST_OVERHEAD;
    bool rowFitsCompute = (computeWorkspace <= ubSize);
    //    (b) dynamic_quant useDb 判定: db 双缓冲单行能放进 UB (无 smooth, groupNum=0, 非 large-shape)
    uint64_t dbPerRow = static_cast<uint64_t>(ubFactor) * DB_UB_COEFF;
    bool useDb = (dbPerRow < ubSize);
    //    (c) 批量聚合写 GM 要求 numCol 块对齐 (int8 需 32 字节 = 32 元素; half/bf16 需 32 字节 = 16 元素, 取 32 兼容 int8)
    bool numColBlockAligned = (numCol % INT8_ALIGN_NUM == 0);

    bool primary = rowFitsCompute && useDb && numColBlockAligned;

    uint32_t rmsMode = primary ? MODE_NORMAL : MODE_SINGLE_N;
    uint32_t quantKey = primary ? GetQuantDbKey(dataType) : GetQuantNonDbKey(dataType);
    uint32_t rmsKey = dtypeKey * 10 + rmsMode;            // normal: 10/30, single_n: 13/33
    uint32_t tilingKey = rmsKey * 100 + quantKey;         // 组合编码

    // 8. 核心分配 + rowBatchSize
    uint32_t headCoreNum = 0, rowPerHeadCore = 0, rowPerTailCore = 0, useCoreNum = 0;
    uint32_t blockFactor = 0, latsBlockFactor = 0;
    uint32_t rowFactor = 1, rowLoop = 0, rowTail = 0, lastBlockRowLoop = 0, lastBlockRowTail = 0;
    uint32_t rowBatchSize = 1;

    if (primary) {
        // MODE_NORMAL: block 分配 (同 add_rms_norm_bias normal)
        blockFactor = 1U;
        uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
        blockFactor *= tileNum;
        useCoreNum = CeilDiv(numRow, blockFactor);
        latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);
        rowFactor = 1;
        rowLoop = blockFactor;
        rowTail = 1;
        lastBlockRowLoop = latsBlockFactor;
        lastBlockRowTail = 1;

        // 【关键】rowBatchSize 动态计算: 用 UB 剩余空间 (减去单行计算工作集) 承载尽可能多的行暂存
        // 每行暂存 = x(ubFactor*sizeofT) + yQuant(outAlignLen) + scale(sizeof(float))
        uint64_t stagingPerRow =
            static_cast<uint64_t>(ubFactor) * sizeofT + static_cast<uint64_t>(outAlignLen) + sizeof(float);
        uint64_t remainUB = (ubSize > computeWorkspace) ? (ubSize - computeWorkspace) : 0u;
        uint32_t maxByUb = (stagingPerRow > 0) ? static_cast<uint32_t>(remainUB / stagingPerRow) : 0u;
        // 上限: MAX_ROW_BATCH (指令重复数) 与 blockFactor (每核最大行数)
        uint32_t cap = std::min<uint32_t>(MAX_ROW_BATCH, blockFactor);
        rowBatchSize = std::min(maxByUb, cap);
        if (rowBatchSize == 0) { rowBatchSize = 1; }  // 至少 1 行
    } else {
        // MODE_SINGLE_N: head/tail 分配, 1 行/批 (超大行场景)
        CalculateMultiCoreDistribution(numRow, numCore, headCoreNum, rowPerHeadCore, rowPerTailCore, useCoreNum);
        blockFactor = rowPerHeadCore;
        latsBlockFactor = rowPerTailCore;
        rowFactor = 1;
        rowLoop = 1;
        rowTail = 1;
        lastBlockRowLoop = 1;
        lastBlockRowTail = 1;
        rowBatchSize = 1;
    }

    context->SetTilingKey(tilingKey);

    // 9. AG 属性
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

    // 10. 填充 TilingData
    AddRmsNormBiasDynamicQuantAGTilingData* tilingData =
        context->GetTilingData<AddRmsNormBiasDynamicQuantAGTilingData>();

    tilingData->groupSize        = *groupSizePtr;
    tilingData->rowLen           = rowLen;
    tilingData->rowTotalNum      = rowTotalNum;
    tilingData->numRow           = numRow;
    tilingData->numCol           = numCol;
    tilingData->epsilon          = epsilon;
    tilingData->avgFactor        = (numCol == 0) ? 0.0f : (1.0f / static_cast<float>(numCol));
    tilingData->dstType          = static_cast<uint32_t>(DT_INT8);
    tilingData->coreNum          = useCoreNum;
    tilingData->headCoreNum      = headCoreNum;
    tilingData->rowPerHeadCore   = rowPerHeadCore;
    tilingData->rowPerTailCore   = rowPerTailCore;
    tilingData->multiRowNum      = 1;
    tilingData->ubFactor         = ubFactor;
    tilingData->blockFactor      = blockFactor;
    tilingData->latsBlockFactor  = latsBlockFactor;
    tilingData->rowFactor        = rowFactor;
    tilingData->rowLoop          = rowLoop;
    tilingData->rowTail          = rowTail;
    tilingData->lastBlockRowLoop = lastBlockRowLoop;
    tilingData->lastBlockRowTail = lastBlockRowTail;
    tilingData->numColAlign      = numColAlign;
    tilingData->hasX2            = (context->GetOptionalInputDesc(IDX_X2) != nullptr) ? 1 : 0;
    tilingData->hasBias          = (context->GetOptionalInputDesc(IDX_BIAS) != nullptr) ? 1 : 0;
    tilingData->rowBatchSize     = rowBatchSize;

    // 11. MC2 AlltoAll 通信配置 (同旧融合算子)
    uint32_t opType = 8;  // batch write
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    std::string algConfig;
    if (aivNum < 40) {
        algConfig = "AlltoAll=level0:fullmesh";
    } else {
        algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    }
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    // 12. BlockDim (必须 >= groupSize 以容纳 AG 阶段)
    uint32_t usedcore = std::max(useCoreNum, (uint32_t)*groupSizePtr);
    context->SetBlockDim(usedcore);

    // 13. Workspace
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = USR_WORKSPACE + sysWorkspaceSize;

    // 14. 日志
    OPS_LOG_I(context->GetNodeName(), "Tiling Key: %u (rmsKey=%u, quantKey=%u, primary=%d)",
              tilingKey, rmsKey, quantKey, (int)primary);
    OPS_LOG_I(context->GetNodeName(), "Block Dim: %u (useCore: %u, groupSize: %u)", usedcore, useCoreNum, *groupSizePtr);
    OPS_LOG_I(context->GetNodeName(), "numRow: %u, numCol: %u, ubFactor: %u, outAlignLen: %u",
              numRow, numCol, ubFactor, outAlignLen);
    OPS_LOG_I(context->GetNodeName(), "rowBatchSize: %u, blockFactor: %u, latsBlockFactor: %u",
              rowBatchSize, blockFactor, latsBlockFactor);
    OPS_LOG_I(context->GetNodeName(), "rowLen: %llu, rowTotalNum: %llu, groupSize: %llu",
              rowLen, rowTotalNum, (uint64_t)*groupSizePtr);
    OPS_LOG_I(context->GetNodeName(), "epsilon: %f, avgFactor: %f", epsilon, tilingData->avgFactor);
    OPS_LOG_I(context->GetNodeName(), "Exit TilingAddRmsNormBiasDynamicQuantAG");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormBiasDynamicQuantAG)
    .Tiling(TilingAddRmsNormBiasDynamicQuantAG)
    .TilingParse<AddRmsNormBiasDynamicQuantAGCompileInfo>(TilingPrepareAddRmsNormBiasDynamicQuantAG);

} // namespace optiling
