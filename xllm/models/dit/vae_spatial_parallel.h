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

#include <glog/logging.h>
#include <torch/torch.h>

#include <vector>

#include "core/framework/parallel_state/process_group.h"

namespace xllm {
namespace dit {

class VaeSpatialParallel {
 public:
  VaeSpatialParallel(int w_split, ProcessGroup* pg, const torch::Device& device)
      : w_split_(w_split),
        pg_(pg),
        device_(device),
        rank_(pg ? pg->rank() : 0) {
    CHECK(w_split_ > 0) << "w_split must be positive";
    CHECK(w_split_ == 1 || pg != nullptr)
        << "ProcessGroup must be provided when w_split > 1";
    if (pg) {
      CHECK(w_split_ == pg->world_size())
          << "w_split must equal ProcessGroup world_size";
    }
  }

  bool is_parallel() const { return w_split_ > 1; }
  int w_split() const { return w_split_; }
  int rank() const { return rank_; }
  int world_size() const { return w_split_; }
  bool has_left() const { return rank_ > 0; }
  bool has_right() const { return rank_ < w_split_ - 1; }

  torch::Tensor split(torch::Tensor x) {
    if (!is_parallel()) return x;
    int64_t width = x.size(-1);
    int64_t base_w = width / w_split_;
    int64_t rem_w = width % w_split_;
    int64_t start_w = rank_ * base_w + std::min<int64_t>(rank_, rem_w);
    int64_t w_local = base_w + (rank_ < rem_w ? 1 : 0);
    VLOG(1) << "[VAE-SP] rank=" << rank_ << " split: global_W=" << width
            << " local_W=" << w_local << " start=" << start_w
            << " shape=" << x.sizes();
    return x
        .index({torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(start_w, start_w + w_local)})
        .contiguous();
  }

  torch::Tensor merge(torch::Tensor local_patch) {
    if (!is_parallel()) return local_patch;

    VLOG(1) << "[VAE-SP] rank=" << rank_
            << " merge: local shape=" << local_patch.sizes();

    // Step 1: all_gather widths (same size for all ranks → safe)
    auto local_shape = torch::tensor(
        {local_patch.size(-2), local_patch.size(-1)},
        torch::TensorOptions().dtype(torch::kInt64).device(device_));
    std::vector<torch::Tensor> shape_list(w_split_);
    for (int i = 0; i < w_split_; ++i) {
      shape_list[i] = torch::empty(
          {2}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    }
    pg_->allgather(local_shape, shape_list);

    int64_t total_w = 0;
    std::vector<int64_t> widths(w_split_);
    std::vector<int64_t> flat_sizes(w_split_);
    int64_t max_flat_size = 0;
    int64_t B = local_patch.size(0), C = local_patch.size(1),
            T = local_patch.size(2), H = local_patch.size(3);
    for (int i = 0; i < w_split_; ++i) {
      widths[i] = shape_list[i][1].item<int64_t>();
      total_w += widths[i];
      flat_sizes[i] = B * C * T * H * widths[i];
      max_flat_size = std::max(max_flat_size, flat_sizes[i]);
    }

    // Step 2: pad to max size and all_gather
    auto local_flat = local_patch.reshape({-1});
    auto padded_flat = torch::zeros({max_flat_size}, local_patch.options());
    padded_flat.index_put_({torch::indexing::Slice(0, local_flat.size(0))},
                           local_flat);

    std::vector<torch::Tensor> gathered_flat(w_split_);
    for (int i = 0; i < w_split_; ++i) {
      gathered_flat[i] = torch::empty({max_flat_size}, local_patch.options());
    }
    pg_->allgather(padded_flat, gathered_flat);

    // Step 3: unpad and reshape per rank (H is uniform in 1D W-only split)
    for (int i = 0; i < w_split_; ++i) {
      gathered_flat[i] = gathered_flat[i]
                             .index({torch::indexing::Slice(0, flat_sizes[i])})
                             .reshape({B, C, T, H, widths[i]});
    }

    // Step 4: assemble
    auto full = torch::empty({B, C, T, H, total_w}, local_patch.options());
    int64_t cur_w = 0;
    for (int i = 0; i < w_split_; ++i) {
      full.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(cur_w, cur_w + widths[i])},
                      gathered_flat[i]);
      cur_w += widths[i];
    }
    VLOG(1) << "[VAE-SP] rank=" << rank_
            << " merge done: global shape=" << full.sizes()
            << " per_rank_widths=[...]";
    return full;
  }

  torch::Tensor exchange(torch::Tensor local_patch, bool pad) {
    if (!is_parallel()) return local_patch;

    auto left_col = local_patch
                        .index({torch::indexing::Slice(),
                                torch::indexing::Slice(),
                                torch::indexing::Slice(),
                                torch::indexing::Slice(),
                                torch::indexing::Slice(0, 1)})
                        .contiguous();
    auto right_col =
        local_patch
            .index({torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(-1, torch::indexing::None)})
            .contiguous();

    // Allocate receive buffers (same shape as neighbor's column).
    torch::Tensor left_recv, right_recv;
    if (has_left()) left_recv = torch::empty_like(right_col);
    if (has_right()) right_recv = torch::empty_like(left_col);

    // Even/odd ordering to avoid deadlock in blocking send/recv ring:
    //   Rank i ⟷ i-1: i sends left_col, receives i-1's right_col as left_recv
    //   Rank i ⟷ i+1: i sends right_col, receives i+1's left_col as right_recv
    // Even ranks send first then recv; odd ranks recv first then send.
    if (rank_ % 2 == 0) {
      if (has_right()) pg_->send(right_col, rank_ + 1);
      if (has_right()) pg_->recv(right_recv, rank_ + 1);
      if (has_left()) pg_->send(left_col, rank_ - 1);
      if (has_left()) pg_->recv(left_recv, rank_ - 1);
    } else {
      if (has_left()) pg_->recv(left_recv, rank_ - 1);
      if (has_left()) pg_->send(left_col, rank_ - 1);
      if (has_right()) pg_->recv(right_recv, rank_ + 1);
      if (has_right()) pg_->send(right_col, rank_ + 1);
    }

    if (pad) {
      auto left_pad = has_left() ? left_recv : torch::zeros_like(left_col);
      auto right_pad = has_right() ? right_recv : torch::zeros_like(right_col);
      return torch::cat({left_pad, local_patch, right_pad}, -1).contiguous();
    } else {
      if (!has_left()) {
        return torch::cat({local_patch, right_recv}, -1).contiguous();
      }
      if (!has_right()) {
        return torch::cat({left_recv, local_patch}, -1).contiguous();
      }
      return torch::cat({left_recv, local_patch, right_recv}, -1).contiguous();
    }
  }

  torch::Tensor gather_full(const torch::Tensor& local) {
    if (!is_parallel()) return local;

    int64_t ndim = local.dim();
    int64_t W_local = local.size(-1);
    auto orig_sizes = local.sizes();

    // prefix_numel = product of all dims except the last (W)
    int64_t prefix_numel = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) prefix_numel *= local.size(d);

    // Step 1: all_gather local widths
    auto local_w = torch::tensor(
        {W_local}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    std::vector<torch::Tensor> w_list(w_split_);
    for (int i = 0; i < w_split_; ++i) {
      w_list[i] = torch::empty(
          {1}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    }
    pg_->allgather(local_w, w_list);

    std::vector<int64_t> widths(w_split_);
    std::vector<int64_t> flat_sizes(w_split_);
    int64_t max_flat_size = 0;
    for (int i = 0; i < w_split_; ++i) {
      widths[i] = w_list[i][0].item<int64_t>();
      flat_sizes[i] = prefix_numel * widths[i];
      max_flat_size = std::max(max_flat_size, flat_sizes[i]);
    }

    // Step 2: pad and all_gather
    auto local_flat = local.reshape({-1});
    auto padded_flat = torch::zeros({max_flat_size}, local.options());
    padded_flat.index_put_({torch::indexing::Slice(0, local_flat.size(0))},
                           local_flat);

    std::vector<torch::Tensor> gathered_flat(w_split_);
    for (int i = 0; i < w_split_; ++i) {
      gathered_flat[i] = torch::empty({max_flat_size}, local.options());
    }
    pg_->allgather(padded_flat, gathered_flat);

    // Step 3: unpad and reshape per rank, then cat along last dim
    std::vector<torch::Tensor> gathered_tensors;
    for (int i = 0; i < w_split_; ++i) {
      std::vector<int64_t> shape(orig_sizes.begin(), orig_sizes.end());
      shape.back() = widths[i];
      gathered_tensors.push_back(
          gathered_flat[i]
              .index({torch::indexing::Slice(0, flat_sizes[i])})
              .reshape(shape));
    }
    return torch::cat(gathered_tensors, -1);
  }

 private:
  int w_split_;
  ProcessGroup* pg_;
  torch::Device device_;
  int rank_;
};

}  // namespace dit
}  // namespace xllm
