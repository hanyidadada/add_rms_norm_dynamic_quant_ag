/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Refer to the License for details.
 */

/*!
 * \file add_rms_norm_bias_dynamic_quant_ag_info.h
 * \brief Host-side compile info struct for AddRmsNormBiasDynamicQuantAG
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_H_

#include <cstdint>
#include "tiling/tiling_api.h"
#include "platform/platform_infos_def.h"
#include "error/ops_error.h"

struct AddRmsNormBiasDynamicQuantAGCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_BIAS_DYNAMIC_QUANT_AG_H_
