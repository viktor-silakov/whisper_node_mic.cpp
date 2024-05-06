#pragma once

#include <string>
#include <vector>
#include "common-sdl.h"
#include "common.h"



// class audio_async;
// class wav_writer;

bool init_audio(audio_async& audio, int capture_id, int sample_rate);
void create_wav_writer(wav_writer& wavWriter, const std::string& filename, int sample_rate);
std::string get_current_datetime();
bool process_audio_non_vad(audio_async& audio, std::vector<float>& pcmf32_new, int n_samples_step, int step_ms);
bool process_audio_vad(audio_async& audio, std::vector<float>& pcmf32_new, std::vector<float>& pcmf32, int sample_rate, int length_ms, float vad_thold, float freq_thold);