/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_AG_ADPT_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_AG_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_dynamic_quant_ag(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& gamma,
    c10::string_view group, int64_t group_size, double epsilon)
{
    auto group_ptr = const_cast<char*>(group.data());

    auto y1_size = x1.sizes().vec();
    y1_size[0] *= group_size;

    std::vector<int64_t> scale_size(y1_size.begin(), y1_size.end() - 1);
    auto x_out_size = x1.sizes();
    auto rstd_size = x1.sizes().vec();
    rstd_size.back() = 1;

    auto y_dtype = at::kChar;
    auto x_out_dtype = x2.scalar_type();
    auto scale_dtype = at::kFloat;
    int64_t dst_type = static_cast<int64_t>(2);

    at::Tensor y1 = at::empty(y1_size, x1.options().dtype(y_dtype));
    at::Tensor scale = at::empty(scale_size, x1.options().dtype(scale_dtype));
    at::Tensor x_out = at::empty(x_out_size, x2.options().dtype(x_out_dtype));
    at::Tensor rstd = at::empty(rstd_size, x1.options().dtype(scale_dtype));

    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantAG, x1, x2, gamma, epsilon, dst_type, group_ptr, group_size, y1, scale, x_out, rstd);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y1, scale, x_out);
}
}
#endif
