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

#include <algorithm>
#include <memory>
#include <string>

#include "autoencoder_kl_wan.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/model_registry.h"
#include "transformer_wan2_2.h"
#include "umt5_encoder.h"
#include "uni_pc_multi_step_scheduler.h"
#include "video_processor.h"
// wsd_test_git111

namespace xllm {

class Wan2_2I2VPipelineImpl : public torch::nn::Module {
 public:
  Wan2_2I2VPipelineImpl(const DiTModelContext& context) {
    options_ = context.get_tensor_options();
    const auto& vae_args = context.get_model_args("vae");
    zdim_ = vae_args.z_dim();
    latents_mean_ = vae_args.vae_latents_mean();
    latents_std_ = vae_args.vae_latents_std();
    LOG(INFO) << "Pipeline loaded latents_mean size=" << latents_mean_.size()
              << " latents_std size=" << latents_std_.size();

    const auto& scheduler_args = context.get_model_args("scheduler");
    num_train_timesteps_ = scheduler_args.num_train_timesteps();

    LOG(INFO) << "Initializing Wan2_2I2V pipeline...";
    vae_ = AutoencoderKLWan(context.get_model_context("vae"));
    transformer_ = Wan22DiTModel(context.get_model_context("transformer"));
    transformer_2_ = Wan22DiTModel(context.get_model_context("transformer_2"));
    umt5_ = UMT5EncoderModel(context.get_model_context("text_encoder"));
    scheduler_ =
        UniPCMultistepScheduler(context.get_model_context("scheduler"));
    video_processor_ = VideoProcessor(context.get_model_context("vae"),
                                      true,
                                      true,
                                      false,
                                      false,
                                      false,
                                      4,
                                      vae_scale_factor_spatial_);
    register_module("vae", vae_);
    register_module("transformer", transformer_);
    register_module("transformer_2", transformer_2_);
    register_module("umt5", umt5_);
    register_module("scheduler", scheduler_);
    register_module("video_processor_", video_processor_);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto images = input.images.defined() ? std::make_optional(input.images)
                                         : std::nullopt;
    auto last_images = input.last_images.defined()
                           ? std::make_optional(input.last_images)
                           : std::nullopt;
    auto prompts = std::make_optional(input.prompts);

    auto negative_prompts = input.negative_prompts.empty()
                                ? std::nullopt
                                : std::make_optional(input.negative_prompts);

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;

    auto image_embeds = input.image_embeds.defined()
                            ? std::make_optional(input.image_embeds)
                            : std::nullopt;

    auto output = forward_impl(images,
                               last_images,
                               prompts,
                               negative_prompts,
                               generation_params.height,
                               generation_params.width,
                               generation_params.num_frames,
                               generation_params.num_inference_steps,
                               generation_params.guidance_scale,
                               generation_params.guidance_scale_2,
                               generation_params.num_videos_per_prompt,
                               seed,
                               latents,
                               image_embeds,
                               prompt_embeds,
                               negative_prompt_embeds,
                               generation_params.max_sequence_length);

    DiTForwardOutput out;
    out.tensors = torch::chunk(output, input.batch_size);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "Wan2_2I2VPipeline loading model from"
              << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto transformer_2_loader = loader->take_component_loader("transformer_2");
    auto vae_loader = loader->take_component_loader("vae");
    auto umt5_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");

    LOG(INFO) << "Wan2_2I2VPipeline model components loaded, start to load "
                 "weights to sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device());
    transformer_2_->load_model(std::move(transformer_2_loader));
    transformer_2_->to(options_.device());
    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device());
    umt5_->load_model(std::move(umt5_loader));
    umt5_->to(options_.device());
    tokenizer_ = tokenizer_loader->tokenizer();
  }

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare_latents(
      torch::Tensor image,
      int64_t batch_size,
      int64_t num_channels_latents = 16,
      int64_t height = 480,
      int64_t width = 832,
      int64_t num_frames = 81,
      std::optional<torch::Tensor> last_image = std::nullopt,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt) {
    LOG(INFO) << "height:" << height << "width:" << width;

    int64_t num_latent_frames =
        (num_frames - 1) / vae_scale_factor_temporal_ + 1;
    int64_t latent_height = height / vae_scale_factor_spatial_;
    int64_t latent_width = width / vae_scale_factor_spatial_;

    LOG(INFO) << "latent_height:" << latent_height
              << "latent_width:" << latent_width;

    std::vector<int64_t> shape = {batch_size,
                                  num_channels_latents,
                                  num_latent_frames,
                                  latent_height,
                                  latent_width};
    torch::Tensor latents_tensor;

    if (latents.has_value()) {
      latents_tensor = latents.value().to(options_.device());
    } else {
      latents_tensor = randn_tensor(shape, seed, options_);
    }

    // wsd
    auto state_dict = StateDictFromSafeTensor::load(
        "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/"
        "wsd_noise.safetensors");
    auto input_random_latents = torch::ones({16, 21, 90, 68}, torch::kFloat32);
    bool is_conv_out_weight_loaded = false;
    weight::load_weight(
        *state_dict, "noise", input_random_latents, is_conv_out_weight_loaded);

    latents_tensor = input_random_latents.unsqueeze(0)
                         .to(options_.device())
                         .to(torch::kFloat32);
    // end

    torch::save(latents_tensor,
                "/home/weinan5/zjs/tensors_save_dir/cpp/latents_tensor_cpp.pt");

    image = image.unsqueeze(2);
    torch::Tensor video_condition;

    if (expand_timesteps_) {
      video_condition = image;
    } else if (!last_image.has_value()) {
      auto zeros = torch::zeros(
          {image.size(0), image.size(1), num_frames - 1, height, width},
          image.options());
      video_condition = torch::cat({image, zeros}, 2);
    } else {
      auto last_img = last_image.value().unsqueeze(2);
      auto zeros = torch::zeros(
          {image.size(0), image.size(1), num_frames - 2, height, width},
          image.options());
      video_condition = torch::cat({image, zeros, last_img}, 2);
    }
    video_condition = video_condition.to(options_.device()).to(torch::kFloat32);
    // save tensor
    // -----------------------------------------------------------------
    torch::save(
        video_condition,
        "/home/weinan5/zjs/tensors_save_dir/cpp/video_condition_cpp.pt");
    // save tensor
    // -----------------------------------------------------------------

    LOG(INFO) << "video_condition shape=" << video_condition.sizes()
              << " device=" << video_condition.device();

    torch::Tensor latents_mean =
        torch::tensor(latents_mean_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(latents_tensor.device(), latents_tensor.dtype());
    torch::Tensor latents_std =
        1.0 / torch::tensor(latents_std_, torch::dtype(torch::kFloat32))
                  .view({1, num_channels_latents, 1, 1, 1})
                  .to(latents_tensor.device(), latents_tensor.dtype());

    torch::Tensor latent_condition;
    try {
      latent_condition = vae_->encode(video_condition).latent_dist.mode();
    } catch (const std::exception& e) {
      LOG(ERROR) << "VAE encode failed: " << e.what()
                 << " video_condition device=" << video_condition.device()
                 << " shape=" << video_condition.sizes();
      throw;
    }

    latent_condition = (latent_condition - latents_mean) * latents_std;

    // save tensor
    // -----------------------------------------------------------------
    torch::save(
        latent_condition,
        "/home/weinan5/zjs/tensors_save_dir/cpp/latent_condition_cpp.pt");
    // save tensor
    // -----------------------------------------------------------------

    if (latent_condition.size(0) == 1 && batch_size > 1) {
      latent_condition = latent_condition.repeat({batch_size, 1, 1, 1, 1});
    }

    if (expand_timesteps_) {
      torch::Tensor first_frame_mask = torch::ones(
          {1, 1, num_latent_frames, latent_height, latent_width},
          options_.dtype(torch::kFloat32).device(options_.device()));
      first_frame_mask.slice(2, 0, 1) = 0;
      return {latents_tensor, latent_condition, first_frame_mask};
    }

    torch::Tensor mask_lat_size =
        torch::ones({batch_size, 1, num_frames, latent_height, latent_width},
                    options_.dtype(torch::kFloat32).device(options_.device()));

    if (!last_image.has_value()) {
      for (int64_t i = 1; i < num_frames; ++i) {
        mask_lat_size.select(2, i).fill_(0);
      }
    } else {
      for (int64_t i = 1; i < num_frames - 1; ++i) {
        mask_lat_size.select(2, i).fill_(0);
      }
    }

    torch::Tensor first_frame_mask = mask_lat_size.slice(2, 0, 1);
    first_frame_mask = torch::repeat_interleave(
        first_frame_mask, vae_scale_factor_temporal_, 2);

    torch::Tensor rest_mask = mask_lat_size.slice(2, 1);
    mask_lat_size = torch::cat({first_frame_mask, rest_mask}, 2);

    mask_lat_size = mask_lat_size.view({batch_size,
                                        -1,
                                        vae_scale_factor_temporal_,
                                        latent_height,
                                        latent_width});
    mask_lat_size = mask_lat_size.transpose(1, 2);

    // wsd
    torch::save(mask_lat_size,
                "/home/weinan5/zjs/tensors_save_dir/cpp/msk_cpp.pt");

    mask_lat_size = mask_lat_size.to(latent_condition.device());

    torch::Tensor combined_condition =
        torch::cat({mask_lat_size, latent_condition}, 1);

    torch::save(
        combined_condition,
        "/home/weinan5/zjs/tensors_save_dir/cpp/combined_condition_cpp.pt");

    return {latents_tensor, combined_condition, first_frame_mask};
  }

  torch::Tensor get_t5_prompt_embeds(std::vector<std::string>& prompt,
                                     int64_t num_videos_per_prompt = 1,
                                     int64_t max_sequence_length = 512) {
    int64_t batch_size = prompt.size();

    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);

    for (int i = 0; i < prompt.size(); i++) {
      LOG(INFO) << "get_t5_prompt_embeds prompt content" << prompt[i];
    }

    CHECK(tokenizer_->batch_encode(prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      ids.resize(max_sequence_length, 0);
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * max_sequence_length);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_sequence_length})
            .to(options_.device());
    // save tensor
    // -----------------------------------------------------------------
    torch::save(input_ids,
                "/home/weinan5/zjs/tensors_save_dir/cpp/input_ids_cpp.pt");
    // save tensor
    // -----------------------------------------------------------------

    torch::Tensor attention_mask =
        (1.0 - (input_ids > 0).to(options_.dtype()).unsqueeze(1).unsqueeze(2)) *
        (std::numeric_limits<float>::lowest());
    torch::Tensor prompt_embeds = umt5_->forward(input_ids, attention_mask);

    prompt_embeds = prompt_embeds.to(options_);

    torch::save(
        prompt_embeds,
        "/home/weinan5/zjs/tensors_save_dir/cpp/prompt_embeds_tensor_cpp.pt");

    auto seq_lens = (input_ids > 0).sum(1).to(torch::kLong);
    LOG(INFO) << "prompt_embeds shape" << prompt_embeds.sizes();
    for (int64_t i = 0; i < batch_size; ++i) {
      LOG(INFO) << "seq_lens[" << i << "] = " << seq_lens[i].item<int64_t>();
    }

    std::vector<torch::Tensor> trimmed_embeds;
    trimmed_embeds.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
      int64_t seq_len = seq_lens[i].item<int64_t>();
      auto trimmed = prompt_embeds[i].slice(0, 0, seq_len);
      int64_t padding_len = max_sequence_length - trimmed.size(0);
      if (padding_len > 0) {
        auto zeros =
            torch::zeros({padding_len, trimmed.size(1)}, trimmed.options());
        trimmed = torch::cat({trimmed, zeros}, 0);
      }
      trimmed_embeds.push_back(trimmed);
    }
    prompt_embeds = torch::stack(trimmed_embeds, 0);

    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_videos_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_videos_per_prompt, seq_len, -1});

    return prompt_embeds;
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<std::vector<std::string>> negative_prompt,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> negative_prompt_embeds,
      bool do_classifier_free_guidance = true,
      int64_t num_videos_per_prompt = 1,
      int64_t max_sequence_length = 226) {
    torch::Tensor prompt_embeds_tensor;
    torch::Tensor negative_prompt_embeds_tensor;

    if (prompt_embeds.has_value()) {
      prompt_embeds_tensor = prompt_embeds.value();
    } else if (prompt.has_value()) {
      prompt_embeds_tensor = get_t5_prompt_embeds(
          prompt.value(), num_videos_per_prompt, max_sequence_length);
    }

    int64_t batch_size;
    if (prompt_embeds.has_value()) {
      batch_size = prompt_embeds_tensor.size(0);
    } else if (prompt.has_value()) {
      batch_size = prompt.value().size();
    }

    if (do_classifier_free_guidance) {
      LOG(INFO) << "test encode_prompt do_classifier_free_guidance";
      if (negative_prompt.has_value()) {
        LOG(INFO) << "test encode_prompt, negative_prompt.has_value";
        int prompt_size = negative_prompt.value().size();
        LOG(INFO) << "negative_prompt.size()" << prompt_size;
        for (int i = 0; i < prompt_size; i++) {
          LOG(INFO) << "negative prompt" << negative_prompt.value()[i];
        }
        negative_prompt_embeds_tensor =
            get_t5_prompt_embeds(negative_prompt.value(),
                                 num_videos_per_prompt,
                                 max_sequence_length);
      } else if (negative_prompt_embeds.has_value()) {
        negative_prompt_embeds_tensor = negative_prompt_embeds.value();
      } else {
        std::vector<std::string> empty_prompt = {""};
        negative_prompt_embeds_tensor = get_t5_prompt_embeds(
            empty_prompt, num_videos_per_prompt, max_sequence_length);
      }
    }
    // save tensor
    // -----------------------------------------------------------------
    LOG(INFO) << "prompt_embeds_tensor shape" << prompt_embeds_tensor.sizes();
    LOG(INFO) << "negative prompt_embeds_tensor shape"
              << negative_prompt_embeds_tensor.sizes();

    // auto save_prompt_embeds = prompt_embeds_tensor.to(torch::kCPU);
    // auto save_negative_prompt_embeds =
    // negative_prompt_embeds_tensor.to(torch::kCPU);
    // torch::save(prompt_embeds_tensor,
    // "/home/weinan5/zjs/tensors_save_dir/cpp/prompt_embeds_tensor_cpp.pt");
    // torch::save(negative_prompt_embeds_tensor,
    // "/home/weinan5/zjs/tensors_save_dir/cpp/negative_prompt_embeds_tensor_cpp.pt");

    // save tensor
    // -----------------------------------------------------------------

    return {prompt_embeds_tensor, negative_prompt_embeds_tensor};
  }

  torch::Tensor forward_impl(
      std::optional<torch::Tensor> images = std::nullopt,
      std::optional<torch::Tensor> last_images = std::nullopt,
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      int64_t height = 512,
      int64_t width = 512,
      int64_t num_frames = 16,
      int64_t num_inference_steps = 28,
      float guidance_scale = 5.0f,
      float guidance_scale_2 = -1.0f,
      int64_t num_videos_per_prompt = 1,
      int64_t seed = 42,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> image_embeds = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else if (prompt_embeds.has_value()) {
      batch_size = prompt_embeds.value().size(0);
    }

    int64_t total_batch_size = batch_size * num_videos_per_prompt;
    bool has_neg_prompt =
        negative_prompt.has_value() || (negative_prompt_embeds.has_value());

    bool do_classifier_free_guidance = guidance_scale > 1.0f;

    LOG(INFO) << "num_frames 的值为：" << num_frames;

    if (num_frames % vae_scale_factor_temporal_ != 1) {
      LOG(WARNING) << "num_frames - 1 has to be divisible by "
                   << vae_scale_factor_temporal_
                   << ". Rounding to the nearest number.";
      num_frames =
          num_frames / vae_scale_factor_temporal_ * vae_scale_factor_temporal_ +
          1;
    }
    num_frames = std::max(num_frames, static_cast<int64_t>(1));

    int64_t patch_size_h = transformer_->patch_size()[1];
    int64_t patch_size_w = transformer_->patch_size()[2];

    int64_t dw = vae_scale_factor_spatial_ * patch_size_w;
    int64_t dh = vae_scale_factor_spatial_ * patch_size_h;
    int64_t max_area = height * width;  // size param treated as area constraint

    // Get actual image dimensions to derive aspect ratio
    int64_t ih = images.has_value() ? images.value().size(-2) : height;
    int64_t iw = images.has_value() ? images.value().size(-1) : width;
    double ratio = static_cast<double>(iw) / ih;

    // Candidate 1: floor width first
    int64_t ow1 = static_cast<int64_t>(std::sqrt(max_area * ratio)) / dw * dw;
    int64_t oh1 = (max_area / ow1) / dh * dh;
    double ratio1 = static_cast<double>(ow1) / oh1;

    // Candidate 2: floor height first
    int64_t oh2 = static_cast<int64_t>(std::sqrt(max_area / ratio)) / dh * dh;
    int64_t ow2 = (max_area / oh2) / dw * dw;
    double ratio2 = static_cast<double>(ow2) / oh2;

    // Pick the one that preserves aspect ratio better
    int64_t calc_height, calc_width;
    if (std::max(ratio / ratio1, ratio1 / ratio) <
        std::max(ratio / ratio2, ratio2 / ratio)) {
      calc_width = ow1;
      calc_height = oh1;
    } else {
      calc_width = ow2;
      calc_height = oh2;
    }

    /*
    int64_t h_multiple_of = vae_scale_factor_spatial_ * patch_size_h;
    int64_t w_multiple_of = vae_scale_factor_spatial_ * patch_size_w;
    // new add below 3 lines, should be a optim from mindiesd
    double aspect_ratio = static_cast<double>(height) / width;
    int64_t target_h = static_cast<int64_t>(std::sqrt(height * width *
    aspect_ratio)); int64_t target_w = static_cast<int64_t>(std::sqrt(height *
    width / aspect_ratio));

    LOG(INFO) << "target_h " << target_h << "target_w" << target_w;

    // change from height 2 target_h
    int64_t calc_height = target_h / h_multiple_of * h_multiple_of;
    int64_t calc_width = target_w / w_multiple_of * w_multiple_of;

    LOG(INFO) << "calc_height " << calc_height << "calc_height" << calc_height;
    */
    if (height != calc_height || width != calc_width) {
      // LOG(WARNING) << "height and width must be multiples of (" <<
      // h_multiple_of
      //              << ", " << w_multiple_of
      //              << ") for proper patchification. Adjusting (" << height
      //              << ", " << width << ") -> (" << calc_height << ", "
      //              << calc_width << ").";
      height = calc_height;
      width = calc_width;
    }

    if (boundary_ratio_ > 0.0f && guidance_scale_2 < 0.0f) {
      guidance_scale_2 = guidance_scale;
    }

    // auto state_dict_tensor = StateDictFromSafeTensor::load(
    //         "/home/weinan5/zjs/tensors_save_dir/latents_0_999.safetensors");
    // auto input_prompt_model_input = torch::ones({1, 36, 90, 68, 21},
    // torch::kFloat32); bool is_conv_out_weight_loaded_3 = false;
    //  weight::load_weight(*state_dict_3, "latents", input_prompt_model_input,
    //  is_conv_out_weight_loaded_3); prompt =
    //  input_prompt_model_input.to(options_.device()).to(prepared_latents.dtype());

    auto [encoded_prompt_embeds, encoded_negative_embeds] =
        encode_prompt(prompt,
                      negative_prompt,
                      prompt_embeds,
                      negative_prompt_embeds,
                      do_classifier_free_guidance,
                      num_videos_per_prompt,
                      max_sequence_length);
    /*
    auto a = encoded_prompt_embeds;
    auto state_dict_1 = StateDictFromSafeTensor::load(
                    "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/prompt_embeds.safetensors");
    auto input_prompt_embeds = torch::ones({1, 512, 4096}, torch::kFloat32);
    bool is_conv_out_weight_loaded_1 = false;
    weight::load_weight(*state_dict_1, "saved_111", input_prompt_embeds,
    is_conv_out_weight_loaded_1);

    encoded_prompt_embeds =
    input_prompt_embeds.to(options_.device()).to(a.dtype());



    auto state_dict_2 = StateDictFromSafeTensor::load(
                    "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/negative_prompt_embeds.safetensors");
    auto input_negative_prompt_embeds = torch::ones({1, 512, 4096},
    torch::kFloat32); bool is_conv_out_weight_loaded_2 = false;
    weight::load_weight(*state_dict_2, "saved_222",
    input_negative_prompt_embeds, is_conv_out_weight_loaded_2);

    encoded_negative_embeds =
    input_negative_prompt_embeds.to(options_.device()).to(a.dtype());

    torch::save(encoded_prompt_embeds,
    "/home/weinan5/zjs/tensors_save_dir/cpp/prompt_embeds_tensor_cpp.pt");
    torch::save(encoded_negative_embeds,
    "/home/weinan5/zjs/tensors_save_dir/cpp/negative_prompt_embeds_tensor_cpp.pt");
    */

    scheduler_->set_timesteps(num_inference_steps, options_.device());
    torch::Tensor timesteps = scheduler_->timesteps();

    int64_t num_channels_latents = zdim_;
    torch::Tensor input_image;

    if (images.has_value()) {
      input_image = images.value();
    } else {
      // No input image provided — create a blank white image (C=3, H, W)
      LOG(WARNING) << "No input image provided for I2V pipeline. "
                   << "Using blank white image as fallback.";
      input_image = torch::ones({3, height, width}, torch::kFloat32);
    }
    LOG(INFO) << "input_image张量形状: " << input_image.sizes();
    torch::Tensor preprocessed_image =
        video_processor_->preprocess(input_image, height, width);
    preprocessed_image =
        preprocessed_image.to(options_.device(), torch::kFloat32);

    torch::save(
        preprocessed_image,
        "/home/weinan5/zjs/tensors_save_dir/cpp/image_preprocess_cpp.pt");

    // Ensure 4D: (B, C, H, W) — prepare_latents will add time dim
    if (preprocessed_image.dim() == 3) {
      // (C, H, W) -> (1, C, H, W)
      preprocessed_image = preprocessed_image.unsqueeze(0);
    }

    std::optional<torch::Tensor> preprocessed_last_image;
    if (last_images.has_value()) {
      torch::Tensor last_img =
          video_processor_->preprocess(last_images.value(), height, width);
      last_img = last_img.to(options_.device(), torch::kFloat32);
      if (last_img.dim() == 3) {
        last_img = last_img.unsqueeze(0);
      }
      preprocessed_last_image = last_img;
    }

    torch::Tensor prepared_latents, latent_condition, first_frame_mask;
    std::tie(prepared_latents, latent_condition, first_frame_mask) =
        prepare_latents(preprocessed_image,
                        total_batch_size,
                        num_channels_latents,
                        height,
                        width,
                        num_frames,
                        preprocessed_last_image,
                        seed,
                        latents);
    /*
     auto state_dict_4 = StateDictFromSafeTensor::load(
                  "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/latents.safetensors");
     auto input_latents = torch::ones({1, 16, 21, 90, 68}, torch::kFloat32);
     bool is_conv_out_weight_loaded_4 = false;
     weight::load_weight(*state_dict_4, "saved_444", input_latents,
     is_conv_out_weight_loaded_4);

     prepared_latents =
     prepared_latents.to(options_.device()).to(prepared_latents.dtype());


     auto state_dict_5 = StateDictFromSafeTensor::load(
                  "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/condition.safetensors");
     auto input_condition = torch::ones({1, 20, 21, 90, 68}, torch::kFloat32);
     bool is_conv_out_weight_loaded_5 = false;
     weight::load_weight(*state_dict_5, "saved_555", input_condition,
     is_conv_out_weight_loaded_5);

     latent_condition =
     input_condition.to(options_.device()).to(prepared_latents.dtype());
     */

    float boundary_timestep =
        boundary_ratio_ > 0.0f ? boundary_ratio_ * num_train_timesteps_ : -1.0f;

    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i];
      int64_t total_steps = timesteps.numel();
      bool should_save = (i == 0 || i == 1 || i == 2 || i == total_steps - 1);
      auto save_tensor = [&](const torch::Tensor& tensor,
                             const std::string& name) {
        if (should_save) {
          torch::save(tensor.contiguous(),
                      "/home/weinan5/zjs/tensors_save_dir/cpp/" + name + "_t" +
                          std::to_string(i) + "_cpp.pt");
        }
      };

      Wan22DiTModel current_model = nullptr;
      float current_guidance;

      if (boundary_timestep < 0 || t.item<float>() >= boundary_timestep) {
        LOG(INFO) << "high-noise t:" << t << "boundary_timestep"
                  << boundary_timestep;
        current_model = transformer_;
        current_guidance = guidance_scale;
      } else {
        LOG(INFO) << "low-noise t:" << t << "boundary_timestep"
                  << boundary_timestep;
        current_model = transformer_2_;
        current_guidance = guidance_scale_2;
      }

      torch::Tensor latent_model_input;
      torch::Tensor timestep_input;

      if (expand_timesteps_) {
        latent_model_input = (1 - first_frame_mask) * latent_condition +
                             first_frame_mask * prepared_latents;
        latent_model_input = latent_model_input.to(prepared_latents.dtype());

        torch::Tensor temp_ts = (first_frame_mask[0][0]
                                     .slice(1, 0, first_frame_mask.size(2), 2)
                                     .slice(2, 0, first_frame_mask.size(3), 2) *
                                 t)
                                    .flatten();
        timestep_input =
            temp_ts.unsqueeze(0).expand({prepared_latents.size(0), -1});
      } else {
        latent_model_input =
            torch::cat({prepared_latents, latent_condition}, 1);
        latent_model_input = latent_model_input.to(prepared_latents.dtype());

        if (i == timesteps.numel() - 1) {
          // save tensor
          // -----------------------------------------------------------------
          torch::save(latent_model_input,
                      "/home/weinan5/zjs/tensors_save_dir/cpp/"
                      "latent_model_input_cpp.pt");
        }
        // save tensor
        // -----------------------------------------------------------------
        save_tensor(prepared_latents, "latent_input");

        // std::string save_path =
        // "/home/weinan5/zjs/tensors_save_dir/cpp/latent_model_input" +
        // std::to_string(i) + "_cpp.pt"; torch::save(latent_model_input,
        // save_path);

        /*
        auto state_dict_3 = StateDictFromSafeTensor::load(
                   "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/latent_model_input.safetensors");
        auto input_latent_model_input = torch::ones({1, 36, 21, 90, 68},
        torch::kFloat32); bool is_conv_out_weight_loaded_3 = false;
        weight::load_weight(*state_dict_3, "saved_333",
        input_latent_model_input, is_conv_out_weight_loaded_3);

        latent_model_input =
        input_latent_model_input.to(options_.device()).to(prepared_latents.dtype());;
        */
        timestep_input = t.expand(prepared_latents.size(0));
      }

      // [PYTHON_INPUT] DISABLED: Python/C++ tensor shape 不匹配导致精度恶化
      // - encoded_prompt_embeds: Python [seq_len,4096] vs C++
      // [1,max_seq_len,4096]
      // - patch_embedding: Python 有 seq_len padding，C++ 没有
      // 只保留 flow_shift=5.0 修复（在 uni_pc_multi_step_scheduler.h 中）
      // {
      //   auto sd = StateDictFromSafeTensor::load(
      //       "/home/weinan5/zjs/tensors_save_dir/saved_safetensors/dit_inputs.safetensors");
      //   auto ts = sd->get_tensor("timestep_input");
      //   if (ts.defined()) {
      //     timestep_input =
      //     ts.to(options_.device()).to(prepared_latents.dtype());
      //   }
      //   auto ctx = sd->get_tensor("encoded_prompt_embeds");
      //   if (ctx.defined()) {
      //     encoded_prompt_embeds =
      //     ctx.to(options_.device()).to(prepared_latents.dtype());
      //   }
      // }

      torch::Tensor noise_pred = current_model->forward(false,
                                                        i,
                                                        latent_model_input,
                                                        timestep_input,
                                                        encoded_prompt_embeds,
                                                        torch::Tensor());
      if (i == timesteps.numel() - 1) {
        torch::save(noise_pred,
                    "/home/weinan5/zjs/tensors_save_dir/cpp/noise_pred_cpp.pt");
      }
      save_tensor(noise_pred, "noise_pred");

      if (do_classifier_free_guidance) {
        LOG(INFO) << "do the cfg logic";
        LOG(INFO) << "[DIAG_CFG] before uncond forward, latent dtype="
                  << latent_model_input.dtype()
                  << " timestep dtype=" << timestep_input.dtype()
                  << " neg_embeds dtype=" << encoded_negative_embeds.dtype()
                  << " neg_embeds shape=" << encoded_negative_embeds.sizes();
        torch::Tensor noise_uncond =
            current_model->forward(true,
                                   i,
                                   latent_model_input,
                                   timestep_input,
                                   encoded_negative_embeds,
                                   torch::Tensor());
        if (i == timesteps.numel() - 1) {
          torch::save(
              noise_uncond,
              "/home/weinan5/zjs/tensors_save_dir/cpp/noise_uncond_cpp.pt");
        }
        save_tensor(noise_uncond, "noise_uncond");

        // CFG in FP32 to avoid BF16 amplification by guidance_scale
        auto noise_pred_f = noise_pred.to(torch::kFloat32);
        auto noise_uncond_f = noise_uncond.to(torch::kFloat32);
        noise_pred = (noise_uncond_f + static_cast<float>(current_guidance) *
                                           (noise_pred_f - noise_uncond_f))
                         .to(noise_pred.dtype());
        noise_uncond.reset();
        if (i == timesteps.numel() - 1) {
          torch::save(noise_pred,
                      "/home/weinan5/zjs/tensors_save_dir/cpp/"
                      "noise_pred_with_cfg_cpp.pt");
        }
        save_tensor(noise_pred, "noise_pred_with_cfg");
      }

      LOG(INFO) << "t=" << t.item<float>();
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      if (i == timesteps.numel() - 1) {
        torch::save(prev_latents,
                    "/home/weinan5/zjs/tensors_save_dir/cpp/"
                    "after_scheduler_step_cpp.pt");
      }
      save_tensor(prev_latents, "after_scheduler_step");

      prepared_latents = prev_latents.detach().to(options_.dtype());
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }

    if (expand_timesteps_) {
      prepared_latents = (1 - first_frame_mask) * latent_condition +
                         first_frame_mask * prepared_latents;
    }

    torch::Tensor video;
    prepared_latents = prepared_latents.to(options_.dtype());

    torch::Tensor latents_mean =
        torch::tensor(latents_mean_, torch::dtype(torch::kFloat32))
            .view({1, num_channels_latents, 1, 1, 1})
            .to(prepared_latents.device(), prepared_latents.dtype());
    torch::Tensor latents_std =
        1.0 / torch::tensor(latents_std_, torch::dtype(torch::kFloat32))
                  .view({1, num_channels_latents, 1, 1, 1})
                  .to(prepared_latents.device(), prepared_latents.dtype());

    prepared_latents = prepared_latents / latents_std + latents_mean;
    LOG(INFO) << "________________________开始decode_________________________";
    torch::save(
        prepared_latents,
        "/home/weinan5/zjs/tensors_save_dir/cpp/denoised_latents_cpp.pt");
    video = vae_->decode(prepared_latents.to(torch::kFloat32)).sample;
    torch::save(video,
                "/home/weinan5/zjs/tensors_save_dir/cpp/vae_decode_cpp.pt");
    LOG(INFO) << "________________________结束decode_________________________";
    LOG(INFO) << "输出张量video的shape是：" << video.sizes();
    // save tensor -----------------------------------------------------------
    torch::save(video, "/home/weinan5/zjs/tensors_save_dir/cpp/video_cpp.pt");
    // save tensor ----------------------------------------------------------
    video = video_processor_->postprocess_video(video);

    return video;
  }

 private:
  UniPCMultistepScheduler scheduler_{nullptr};
  AutoencoderKLWan vae_{nullptr};
  Wan22DiTModel transformer_{nullptr};
  Wan22DiTModel transformer_2_{nullptr};
  UMT5EncoderModel umt5_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_{nullptr};
  VideoProcessor video_processor_{nullptr};

  float vae_scaling_factor_;
  float vae_shift_factor_;
  int64_t vae_scale_factor_spatial_ = 8;
  int64_t vae_scale_factor_temporal_ = 4;
  float boundary_ratio_ = 0.9f;
  bool expand_timesteps_ = false;
  int64_t zdim_ = 16;
  float num_train_timesteps_ = 1000.0f;
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
  torch::TensorOptions options_;
};
TORCH_MODULE(Wan2_2I2VPipeline);

REGISTER_DIT_MODEL(wan2_2, Wan2_2I2VPipeline);

}  // namespace xllm
