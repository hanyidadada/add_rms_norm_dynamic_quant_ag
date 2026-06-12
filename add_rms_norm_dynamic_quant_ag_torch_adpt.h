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
std::tuple<at::Tensor,at::Tensor,at::Tensor> add_rms_norm_dynamic_quant_ag(const at::Tensor & y1, const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, c10::string_view group, int64_t group_size, double epsilon)
{
    ::std::array<bool,2> output_mask = {true, false};
    const c10::optional<at::Tensor> smooth_scale1 = c10::nullopt;
    const c10::optional<at::Tensor> smooth_scale2 = c10::nullopt;
    const c10::optional<at::Tensor> beta = c10::nullopt;
    auto group_ptr = const_cast<char *>(group.data());

    // std::vector<int64_t> y1_size(x1.sizes().vec());
    // y1_size[0] *= group_size;

    auto y2_size = at::IntArrayRef{};
    auto x_out_size = x1.sizes();
    std::vector<int64_t> scale1_size(x1.sizes().vec());
    scale1_size[0] *= group_size;
    scale1_size.pop_back();
    auto scale2_size = std::vector<int64_t>{};
    auto y_dtype = at::kChar;
    auto x_out_dtype = x2.scalar_type();
    auto scale_dtype = at::kFloat;
    // at::Tensor y1 = at::empty(y1_size, x1.options().dtype(y_dtype));

    at::Tensor y2 = at::empty(y2_size, x1.options().dtype(y_dtype));
    at::Tensor x_out = at::empty(x_out_size, x2.options().dtype(x_out_dtype));
    at::Tensor scale1 = at::empty(scale1_size, x1.options().dtype(scale_dtype));
    at::Tensor scale2 = at::empty(scale2_size, x1.options().dtype(scale_dtype));
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantAG, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, group_ptr, group_size, epsilon, output_mask, y1, y2, x_out, scale1, scale2);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y1, x_out, scale1);
}
}
#endif