#pragma once

#include "whisper.h"
#include "params.h"

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

struct whisper_context* init_whisper_context(const whisper_params& params, int argc, char** argv);
void print_processing_info(const whisper_context* ctx, const whisper_params& params, int n_samples_step, int n_samples_len, int n_samples_keep, bool use_vad, int n_new_line);
bool open_output_file(std::ofstream& fout, const std::string& filename);
void print_whisper_result(whisper_context* ctx, const whisper_params& params, bool use_vad, int n_iter, const std::chrono::high_resolution_clock::time_point& t_start, std::ofstream& fout);
void update_prompt_tokens(whisper_context* ctx, std::vector<whisper_token>& prompt_tokens, bool no_context);