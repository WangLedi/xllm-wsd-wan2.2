/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "core/layers/common/quant_linear_helpers.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm::dit {

// Supported linear parallelism strategies.
enum class LinearType {
  // No parallelism — single-device matmul via AddMatmulWeightTransposed.
  Default,
  // Ulysses-style sequence parallelism with all2all communication.
  SequenceParallel,
  // Megatron-style tensor parallelism with column/row splitting +
  // gather/reduce.
  TensorParallel,
};

// ── Sequence Parallel Options ──────────────────────────────────────────────
//
// Configures all2all communication for Ulysses-style sequence parallelism.
//
// If before_attention=true:  linear → all2all (scatter heads, gather seq)
// If before_attention=false: all2all → linear (gather heads, scatter seq)
struct SpOptions {
  int64_t head_num = 0;
  int64_t head_dim = 0;
  int64_t hidden_size = 0;
  bool before_attention = false;
  ProcessGroup* process_group = nullptr;

  SpOptions() = default;

  SpOptions(int64_t head_num,
            int64_t head_dim,
            int64_t hidden_size,
            bool before_attention,
            ProcessGroup* process_group = nullptr)
      : head_num(head_num),
        head_dim(head_dim),
        hidden_size(hidden_size),
        before_attention(before_attention),
        process_group(process_group) {}

  void validate() const {
    CHECK(head_num > 0) << "SpOptions: head_num must be > 0, got " << head_num;
    CHECK(head_dim > 0) << "SpOptions: head_dim must be > 0, got " << head_dim;
    CHECK(hidden_size > 0) << "SpOptions: hidden_size must be > 0, got "
                           << hidden_size;
    CHECK(hidden_size == head_dim * head_num)
        << "SpOptions: hidden_size (" << hidden_size
        << ") must equal head_dim (" << head_dim << ") * head_num (" << head_num
        << ")";
    if (!process_group) {
      LOG(ERROR) << "SpOptions: process_group is nullptr — "
                 << "all2all communication requires a valid process group";
    }
  }
};

// ── Tensor Parallel Options ────────────────────────────────────────────────
//
// Configures Megatron-style tensor parallelism.
//
// Column parallel: weight is split along dim 0 (output features).
//   Each rank holds out_features/tp_size rows. Optionally gathers output.
//
// Row parallel: weight is split along dim 1 (input features).
//   Each rank holds in_features/tp_size columns. Reduces output across ranks.
struct TpOptions {
  bool column_parallel = true;
  int64_t tp_rank = 0;
  int64_t tp_size = 1;
  bool gather_output = false;
  bool need_scatter = false;
  ProcessGroup* process_group = nullptr;

  TpOptions() = default;

  TpOptions(bool column_parallel,
            int64_t tp_rank,
            int64_t tp_size,
            bool gather_output = false,
            bool need_scatter = false,
            ProcessGroup* process_group = nullptr)
      : column_parallel(column_parallel),
        tp_rank(tp_rank),
        tp_size(tp_size),
        gather_output(gather_output),
        need_scatter(need_scatter),
        process_group(process_group) {}

  void validate() const {
    CHECK(tp_size > 0) << "TpOptions: tp_size must be > 0, got " << tp_size;
    CHECK(tp_rank >= 0 && tp_rank < tp_size)
        << "TpOptions: tp_rank (" << tp_rank
        << ") must be in [0, tp_size=" << tp_size << ")";
    if (!process_group) {
      LOG(ERROR) << "TpOptions: process_group is nullptr — "
                 << "tensor parallel communication requires a valid process "
                    "group";
    }
  }
};

class DiTParallelLinearImpl : public torch::nn::Module {
 public:
  DiTParallelLinearImpl(layer::AddMatmulWeightTransposed linear,
                        const std::string& module_name,
                        LinearType linear_type = LinearType::Default,
                        const SpOptions& sp_options = SpOptions())
      : in_features_(0),
        out_features_(0),
        has_bias_(false),
        linear_type_(linear_type),
        sp_options_(sp_options),
        tp_options_(std::nullopt) {
    linear_ = register_module(module_name, std::move(linear));
    if (linear_type_ == LinearType::SequenceParallel) {
      sp_options_.validate();
    }
  }

  DiTParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const torch::TensorOptions& options,
      LinearType linear_type = LinearType::Default,
      const std::optional<SpOptions>& sp_options = std::nullopt,
      const std::optional<TpOptions>& tp_options = std::nullopt)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias),
        tensor_options_(options),
        linear_type_(linear_type),
        sp_options_(sp_options.value_or(SpOptions())),
        tp_options_(tp_options) {
    switch (linear_type_) {
      case LinearType::Default:
      case LinearType::SequenceParallel:
        linear_ =
            register_module("linear",
                            layer::AddMatmulWeightTransposed(
                                in_features, out_features, bias, options));
        if (linear_type_ == LinearType::SequenceParallel) {
          sp_options_.validate();
        }
        break;
      case LinearType::TensorParallel:
        CHECK(tp_options_.has_value())
            << "DiTParallelLinear: TpOptions required for TensorParallel mode";
        tp_options_->validate();
        init_tp_weights();
        break;
    }
  }

  DiTParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      const torch::TensorOptions& options,
      LinearType linear_type,
      const QuantArgs& quant_args,
      const std::optional<SpOptions>& sp_options = std::nullopt,
      const std::optional<TpOptions>& tp_options = std::nullopt)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias),
        tensor_options_(options),
        linear_type_(linear_type),
        sp_options_(sp_options.value_or(SpOptions())),
        tp_options_(tp_options),
        quant_args_(quant_args),
        output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
    switch (linear_type_) {
      case LinearType::Default:
      case LinearType::SequenceParallel:
        linear_ = register_module(
            "linear",
            layer::AddMatmulWeightTransposed(
                in_features, out_features, bias, options, quant_args_));
        if (linear_type_ == LinearType::SequenceParallel) {
          sp_options_.validate();
        }
        break;
      case LinearType::TensorParallel:
        CHECK(tp_options_.has_value())
            << "DiTParallelLinear: TpOptions required for TensorParallel mode";
        tp_options_->validate();
        init_tp_weights();
        break;
    }
  }

  torch::Tensor forward(const torch::Tensor& input) {
    switch (linear_type_) {
      case LinearType::Default:
        return linear_->forward(input);
      case LinearType::SequenceParallel:
        return forward_sp(input);
      case LinearType::TensorParallel:
        return forward_tp(input);
      default:
        LOG(FATAL) << "DiTParallelLinear: unknown LinearType "
                   << static_cast<int64_t>(linear_type_);
        return input;
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    if (linear_type_ == LinearType::TensorParallel) {
      load_tp_weights(state_dict);
    } else {
      linear_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (linear_type_ == LinearType::TensorParallel) {
      CHECK(tp_weight_loaded_)
          << "DiTParallelLinear: weight not loaded for " << prefix << "weight";
      if (layer::is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
        CHECK(tp_weight_scale_is_loaded_)
            << "DiTParallelLinear: weight_scale not loaded for " << prefix
            << "weight_scale";
      }
      if (has_bias_) {
        CHECK(tp_bias_loaded_)
            << "DiTParallelLinear: bias not loaded for " << prefix << "bias";
      }
    } else {
      linear_->verify_loaded_weights(prefix);
    }
  }

  torch::Tensor get_weight() const {
    return linear_type_ == LinearType::TensorParallel ? tp_weight_
                                                      : torch::Tensor();
  }

 private:
  torch::Tensor forward_sp(const torch::Tensor& input) {
    CHECK(input.dim() == 3)
        << "SP linear expects 3D input {batch, seq, hidden}, got shape "
        << input.sizes();

    const auto group_size = sp_options_.process_group->world_size();

    if (sp_options_.before_attention) {
      auto out = linear_->forward(input);
      auto fn = parallel_state::all_to_all_4D(
          out.view(
              {input.size(0), -1, sp_options_.head_num, sp_options_.head_dim}),
          /*scatter_dim=*/2,
          /*gather_dim=*/1,
          /*async=*/false,
          sp_options_.process_group);
      return fn().view(
          {input.size(0), -1, sp_options_.hidden_size / group_size});
    } else {
      auto fn = parallel_state::all_to_all_4D(
          input.view({input.size(0),
                      -1,
                      sp_options_.head_num / group_size,
                      sp_options_.head_dim}),
          /*scatter_dim=*/1,
          /*gather_dim=*/2,
          /*async=*/false,
          sp_options_.process_group);
      auto gathered = fn().view({input.size(0), -1, sp_options_.hidden_size});
      return linear_->forward(gathered);
    }
  }

  void init_tp_weights() {
    const auto& tp = tp_options_.value();
    auto weight_dtype = quant_args_.quant_descs().empty()
                            ? tensor_options_.dtype()
                            : tensor_options_.dtype(torch::kInt8);
    if (tp.column_parallel) {
      int64_t out_per_partition = out_features_ / tp.tp_size;
      tp_weight_ = register_parameter(
          "weight",
          torch::empty({out_per_partition, in_features_}, weight_dtype),
          /*is_buffer=*/false);
      if (has_bias_) {
        tp_bias_ = register_parameter(
            "bias",
            torch::empty({out_per_partition}, tensor_options_),
            /*is_buffer=*/false);
      }
    } else {
      int64_t in_per_partition = in_features_ / tp.tp_size;
      tp_weight_ = register_parameter(
          "weight",
          torch::empty({out_features_, in_per_partition}, weight_dtype),
          /*is_buffer=*/false);
      if (has_bias_) {
        tp_bias_ =
            register_parameter("bias",
                               torch::empty({out_features_}, tensor_options_),
                               /*is_buffer=*/false);
      }
    }
  }

  torch::Tensor forward_tp(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();
    if (tp.tp_size <= 1) {
      return linear_->forward(input);
    }
    if (layer::is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      return tp.column_parallel ? forward_tp_column_quant(input)
                                : forward_tp_row_quant(input);
    }
    return tp.column_parallel ? forward_tp_column(input)
                              : forward_tp_row(input);
  }

  torch::Tensor forward_tp_column(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();

    auto bias =
        has_bias_ ? std::optional<torch::Tensor>(tp_bias_) : std::nullopt;
    xllm::kernel::MatmulParams params;
    params.a = input;
    params.b = tp_weight_;
    params.bias = bias;
    auto output = xllm::kernel::matmul(params);

    if (tp.gather_output) {
      output = parallel_state::gather(output, tp.process_group, /*dim=*/-1);
    }
    return output;
  }

  torch::Tensor forward_tp_row(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();

    auto x = input;
    if (tp.need_scatter) {
      x = parallel_state::scatter(input, tp.process_group, /*dim=*/-1);
    }

    xllm::kernel::MatmulParams params;
    params.a = x;
    params.b = tp_weight_;
    auto output = xllm::kernel::matmul(params);

    auto orig_dtype = output.dtype();
    auto output_fp32 = output.to(torch::kFloat32);
    output =
        parallel_state::reduce(output_fp32, tp.process_group).to(orig_dtype);

    if (has_bias_) {
      output = output + tp_bias_;
    }
    return output;
  }

  torch::Tensor forward_tp_column_quant(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();
    CHECK(tp_weight_scale_is_loaded_ && tp_weight_scale_.defined())
        << "tp_weight_scale is required for w8a8_dynamic TP column quant";

    auto bias =
        has_bias_ ? std::optional<torch::Tensor>(tp_bias_) : std::nullopt;
    auto w_offset = tp_weight_offset_.defined()
                        ? std::optional<torch::Tensor>(tp_weight_offset_)
                        : std::nullopt;
    auto output = layer::npu_w8a8_dynamic_linear_forward(
        input, tp_weight_, tp_weight_scale_, bias, output_dtype_, w_offset);

    if (tp.gather_output) {
      output = parallel_state::gather(output, tp.process_group, /*dim=*/-1);
    }
    return output;
  }

  torch::Tensor forward_tp_row_quant(const torch::Tensor& input) {
    const auto& tp = tp_options_.value();
    CHECK(tp_weight_scale_is_loaded_ && tp_weight_scale_.defined())
        << "tp_weight_scale is required for w8a8_dynamic TP row quant";

    auto x = input;
    if (tp.need_scatter) {
      x = parallel_state::scatter(input, tp.process_group, /*dim=*/-1);
    }

    auto bias =
        has_bias_ ? std::optional<torch::Tensor>(tp_bias_) : std::nullopt;
    auto w_offset = tp_weight_offset_.defined()
                        ? std::optional<torch::Tensor>(tp_weight_offset_)
                        : std::nullopt;
    auto output = layer::npu_w8a8_dynamic_linear_forward(
        x, tp_weight_, tp_weight_scale_, bias, output_dtype_, w_offset);

    auto orig_dtype = output.dtype();
    auto output_fp32 = output.to(torch::kFloat32);
    output =
        parallel_state::reduce(output_fp32, tp.process_group).to(orig_dtype);

    return output;
  }

  void load_tp_weights(const StateDict& state_dict) {
    const auto& tp = tp_options_.value();

    // Resolve quant method from quant_descs (no-op if quant_args_ is empty).
    layer::resolve_weight_quant_method_for_linear_load(
        quant_args_, state_dict, nullptr, resolved_weight_quant_method_);

    // Lazy quant param registration / fallback to FP16 for TP weights.
    if (layer::is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      if (!tp_weight_scale_.defined()) {
        auto weight_scale_size = tp.column_parallel
                                     ? tp_weight_.size(0)
                                     : tp_weight_.size(0);  // out_features
        std::vector<weight::LazyParameterSpec> specs;
        specs.push_back(
            weight::LazyParameterSpec{&tp_weight_scale_,
                                      &tp_weight_scale_is_loaded_,
                                      "weight_scale",
                                      {weight_scale_size},
                                      tensor_options_.dtype(torch::kFloat32)});
        specs.push_back(
            weight::LazyParameterSpec{&tp_weight_offset_,
                                      &tp_weight_offset_is_loaded_,
                                      "weight_offset",
                                      {weight_scale_size},
                                      tensor_options_.dtype(torch::kFloat32)});
        weight::ensure_parameter_storage(this, specs);
      }
    } else if (!quant_args_.quant_descs().empty()) {
      // Re-register int8 weight back to original dtype for non-quant fallback.
      std::vector<weight::LazyParameterSpec> specs;
      specs.push_back(
          weight::LazyParameterSpec{&tp_weight_,
                                    &tp_weight_loaded_,
                                    "weight",
                                    {tp_weight_.size(0), tp_weight_.size(1)},
                                    tensor_options_});
      weight::ensure_parameter_storage(this, specs);
    }

    if (layer::is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      int64_t weight_axis = tp.column_parallel ? 0 : 1;
      weight::load_sharded_weight(state_dict,
                                  "weight",
                                  weight_axis,
                                  tp.tp_rank,
                                  tp.tp_size,
                                  tp_weight_,
                                  tp_weight_loaded_);
      int64_t scale_axis = 0;
      weight::load_sharded_weight(state_dict,
                                  "weight_scale",
                                  scale_axis,
                                  tp.tp_rank,
                                  tp.tp_size,
                                  tp_weight_scale_,
                                  tp_weight_scale_is_loaded_);
      if (state_dict.has("weight_offset")) {
        weight::load_sharded_weight(state_dict,
                                    "weight_offset",
                                    scale_axis,
                                    tp.tp_rank,
                                    tp.tp_size,
                                    tp_weight_offset_,
                                    tp_weight_offset_is_loaded_);
      }
    } else {
      if (tp.column_parallel) {
        weight::load_sharded_weight(state_dict,
                                    "weight",
                                    /*axis=*/0,
                                    tp.tp_rank,
                                    tp.tp_size,
                                    tp_weight_,
                                    tp_weight_loaded_);
      } else {
        weight::load_sharded_weight(state_dict,
                                    "weight",
                                    /*axis=*/1,
                                    tp.tp_rank,
                                    tp.tp_size,
                                    tp_weight_,
                                    tp_weight_loaded_);
      }
    }

    if (has_bias_) {
      if (tp.column_parallel) {
        weight::load_sharded_weight(state_dict,
                                    "bias",
                                    /*axis=*/0,
                                    tp.tp_rank,
                                    tp.tp_size,
                                    tp_bias_,
                                    tp_bias_loaded_);
      } else {
        weight::load_weight(state_dict, "bias", tp_bias_, tp_bias_loaded_);
      }
    }
  }

  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;
  torch::TensorOptions tensor_options_;
  layer::AddMatmulWeightTransposed linear_{nullptr};
  LinearType linear_type_;
  SpOptions sp_options_;
  std::optional<TpOptions> tp_options_;
  torch::Tensor tp_weight_;
  torch::Tensor tp_bias_;
  torch::Tensor tp_weight_scale_;
  torch::Tensor tp_weight_offset_;
  bool tp_weight_loaded_ = false;
  bool tp_bias_loaded_ = false;
  bool tp_weight_scale_is_loaded_ = false;
  bool tp_weight_offset_is_loaded_ = false;
  QuantArgs quant_args_;
  std::optional<std::string> resolved_weight_quant_method_;
  at::ScalarType output_dtype_;
};

TORCH_MODULE(DiTParallelLinear);

}  // namespace xllm::dit
