/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_request_params.h"

#include "butil/base64.h"
#include "core/common/instance_name.h"
#include "core/common/macros.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "mm_codec.h"
#include "request.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_request_id(const std::string& prefix) {
  return prefix + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::pair<int32_t, int32_t> splitResolution(const std::string& s) {
  size_t pos = s.find('*');
  int32_t width = std::stoi(s.substr(0, pos));
  int32_t height = std::stoi(s.substr(pos + 1));
  return {width, height};
}

// Decode a base64-encoded image string into a torch tensor via OpenCV.
bool decodeBase64Image(const std::string& base64, torch::Tensor& out) {
  std::string raw_bytes;
  if (!butil::Base64Decode(base64, &raw_bytes)) {
    LOG(ERROR) << "Base64 decode failed";
    return false;
  }
  OpenCVImageDecoder decoder;
  if (!decoder.decode(raw_bytes, out)) {
    LOG(ERROR) << "Image decode failed";
    return false;
  }
  return true;
}

void fillImageInputParams(DiTInputParams& input_params,
                          const proto::Input& input) {
  input_params.prompt = input.prompt();
  if (input.has_prompt_2()) {
    input_params.prompt_2 = input.prompt_2();
  }
  if (input.has_negative_prompt()) {
    input_params.negative_prompt = input.negative_prompt();
  }
  if (input.has_negative_prompt_2()) {
    input_params.negative_prompt_2 = input.negative_prompt_2();
  }
  if (input.has_prompt_embed()) {
    input_params.prompt_embed = util::proto_to_torch(input.prompt_embed());
  }
  if (input.has_pooled_prompt_embed()) {
    input_params.pooled_prompt_embed =
        util::proto_to_torch(input.pooled_prompt_embed());
  }
  if (input.has_negative_prompt_embed()) {
    input_params.negative_prompt_embed =
        util::proto_to_torch(input.negative_prompt_embed());
  }
  if (input.has_negative_pooled_prompt_embed()) {
    input_params.negative_pooled_prompt_embed =
        util::proto_to_torch(input.negative_pooled_prompt_embed());
  }
  if (input.has_latent()) {
    input_params.latent = util::proto_to_torch(input.latent());
  }
  if (input.has_masked_image_latent()) {
    input_params.masked_image_latent =
        util::proto_to_torch(input.masked_image_latent());
  }
  if (input.has_mask_image()) {
    decodeBase64Image(input.mask_image(), input_params.mask_image);
  }
  if (input.has_image()) {
    decodeBase64Image(input.image(), input_params.image);
  }
  if (input.has_condition_image()) {
    decodeBase64Image(input.condition_image(), input_params.condition_image);
  }
  if (input.has_control_image()) {
    decodeBase64Image(input.control_image(), input_params.control_image);
  }
}

void fillVideoInputParams(DiTInputParams& input_params,
                          const proto::VideoInput& input) {
  input_params.prompt = input.prompt();
  if (input.has_negative_prompt()) {
    input_params.negative_prompt = input.negative_prompt();
  }
  if (input.has_prompt_embed()) {
    input_params.prompt_embed = util::proto_to_torch(input.prompt_embed());
  }
  if (input.has_negative_prompt_embed()) {
    input_params.negative_prompt_embed =
        util::proto_to_torch(input.negative_prompt_embed());
  }
  if (input.has_image()) {
    decodeBase64Image(input.image(), input_params.image);
  }
  if (input.has_last_image()) {
    decodeBase64Image(input.last_image(), input_params.last_image);
  }
  if (input.has_image_embeds()) {
    input_params.image_embeds = util::proto_to_torch(input.image_embeds());
  }
}

void fillImageGenerationParams(DiTGenerationParams& generation_params,
                               const proto::Parameters& params) {
  if (params.has_size()) {
    auto [w, h] = splitResolution(params.size());
    generation_params.width = w;
    generation_params.height = h;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_num_images_per_prompt()) {
    generation_params.num_images_per_prompt =
        static_cast<uint32_t>(params.num_images_per_prompt());
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_enable_cfg_renorm()) {
    generation_params.enable_cfg_renorm = params.enable_cfg_renorm();
  }
  if (params.has_cfg_renorm_min()) {
    generation_params.cfg_renorm_min = params.cfg_renorm_min();
  }
}

void fillVideoGenerationParams(DiTGenerationParams& generation_params,
                               const proto::VideoParameters& params) {
  if (params.has_size()) {
    auto [w, h] = splitResolution(params.size());
    generation_params.width = w;
    generation_params.height = h;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale_2()) {
    generation_params.guidance_scale_2 = params.guidance_scale_2();
  }
  if (params.has_num_videos_per_prompt()) {
    generation_params.num_videos_per_prompt =
        static_cast<uint32_t>(params.num_videos_per_prompt());
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_num_frames()) {
    generation_params.num_frames = params.num_frames();
  }
  if (params.has_fps()) {
    generation_params.video_fps = params.fps();
  }
  if (params.has_seconds()) {
    generation_params.seconds = params.seconds();
  }
  if (params.has_boundary_ratio()) {
    generation_params.boundary_ratio = params.boundary_ratio();
  }
  if (params.has_flow_shift()) {
    generation_params.flow_shift = params.flow_shift();
  }
}

}  // namespace

DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("imggen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    fillImageInputParams(input_params, request.input());
  }

  if (request.has_parameters()) {
    fillImageGenerationParams(generation_params, request.parameters());
  }
}

DiTRequestParams::DiTRequestParams(const proto::VideoGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("vidgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  generation_params.force_video_output = true;

  if (request.has_input()) {
    fillVideoInputParams(input_params, request.input());
  }

  if (request.has_parameters()) {
    fillVideoGenerationParams(generation_params, request.parameters());
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  if (input_params.prompt.empty()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  return true;
}

}  // namespace xllm
