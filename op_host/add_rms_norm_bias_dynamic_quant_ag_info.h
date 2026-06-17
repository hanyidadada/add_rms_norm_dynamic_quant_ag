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
