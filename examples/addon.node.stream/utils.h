#pragma once

#include "whisper.h"
// #include "params.h"

#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

struct whisper_params {
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  int32_t step_ms = 3000;
  int32_t soft_ms_th = 10000;
  int32_t keep_ms = 200;
  int32_t capture_id = -1;
  int32_t max_tokens = 32;
  int32_t audio_ctx = 0;

  float vad_thold = 0.6f;
  float freq_thold = 100.0f;

  bool speed_up = false;
  bool translate = false;
  bool no_fallback = false;
  bool print_special = false;
  bool no_context = true;
  bool no_timestamps = false;
  bool tinydiarize = false;
  bool save_audio = false;
  bool use_gpu = true;

  std::string language = "en";
  std::string model = "models/ggml-base.en.bin";
  std::string fname_out;
};

struct whisper_context* init_whisper_context(const whisper_params& params,
                                             int argc, char** argv);
void print_processing_info(const whisper_context* ctx,
                           const whisper_params& params, int n_samples_step,
                           int n_samples_len, int n_samples_keep, bool use_vad,
                           int n_new_line);
bool open_output_file(std::ofstream& fout, const std::string& filename);
void print_whisper_result(
    whisper_context* ctx, const whisper_params& params, bool use_vad,
    int n_iter, const std::chrono::high_resolution_clock::time_point& t_start,
    std::ofstream& fout);
void update_prompt_tokens(whisper_context* ctx,
                          std::vector<whisper_token>& prompt_tokens,
                          bool no_context);
void log_debug(const char* func, float energy_all, float energy_last,
               float vad_thold, float freq_thold);

bool save_to_wav(const std::string& filename,
                 const std::vector<float>& audio_data, int sample_rate);