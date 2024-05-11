#include "utils.h"
#include "common-sdl.h"
#include "common.h"

#include <cassert>
#include <cstdio>

void log_debug(const char *func, float energy_all, float energy_last,
                 float vad_thold, float freq_thold) {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    long long milliseconds = value.count();

    auto now_time = std::chrono::system_clock::to_time_t(now);
    char timestamp[24];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now_time));

    fprintf(stderr,
            "[%s.%03lld] %s: energy_all: %f, energy_last: %f, vad_thold: %f, "
            "freq_thold: %f\n",
            timestamp, milliseconds % 1000, func, energy_all, energy_last,
            vad_thold, freq_thold);
  }

struct whisper_context* init_whisper_context(const whisper_params& params, int argc, char** argv) {
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        // whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;

    return whisper_init_from_file_with_params(params.model.c_str(), cparams);
}

void print_processing_info(const whisper_context* ctx, const whisper_params& params, int n_samples_step, int n_samples_len, int n_samples_keep, bool use_vad, int n_new_line) {
    fprintf(stderr, "\n");
    whisper_params local_params = params;
    if (!whisper_is_multilingual(const_cast<whisper_context*>(ctx))) {
        if (local_params.language != "en" || local_params.translate) {
            local_params.language = "en";
            local_params.translate = false;
            fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
        }
    }
    fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
            __func__,
            n_samples_step,
            float(n_samples_step)/WHISPER_SAMPLE_RATE,
            float(n_samples_len )/WHISPER_SAMPLE_RATE,
            float(n_samples_keep)/WHISPER_SAMPLE_RATE,
            local_params.n_threads,
            local_params.language.c_str(),
            local_params.translate ? "translate" : "transcribe",
            local_params.no_timestamps ? 0 : 1);

    if (!use_vad) {
        fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, local_params.no_context);
    } else {
        fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
    }

    fprintf(stderr, "\n");
}

bool open_output_file(std::ofstream& fout, const std::string& filename) {
    fout.open(filename);
    if (!fout.is_open()) {
        fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, filename.c_str());
        return false;
    }
    return true;
}

void print_whisper_result(whisper_context* ctx, const whisper_params& params, bool use_vad, int n_iter, const std::chrono::high_resolution_clock::time_point& t_start, std::ofstream& fout) {
    if (!use_vad) {
        printf("\33[2K\r");
        printf("%s", std::string(100, ' ').c_str());
        printf("\33[2K\r");
    } else {
        const int64_t t1 = (std::chrono::high_resolution_clock::now() - t_start).count()/1000000;
        const int64_t t0 = std::max(static_cast<int64_t>(0), t1 - params.length_ms);
        printf("\n");
        printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
        printf("\n");
    }

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(ctx, i);
        if (params.no_timestamps) {
            printf("%s", text);
            fflush(stdout);
            if (params.fname_out.length() > 0) {
                fout << text;
            }
        } else {
            const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
            std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                output += " [SPEAKER_TURN]";
            }
            output += "\n";
            printf("%s", output.c_str());
            fflush(stdout);
            if (params.fname_out.length() > 0) {
                fout << output;
            }
        }
    }

    if (params.fname_out.length() > 0) {
        fout << std::endl;
    }

    if (use_vad) {
        printf("\n");
        printf("### Transcription %d END\n", n_iter);
    }
}

void update_prompt_tokens(whisper_context* ctx, std::vector<whisper_token>& prompt_tokens, bool no_context) {
    if (!no_context) {
        prompt_tokens.clear();
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const int token_count = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < token_count; ++j) {
                prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
            }
        }
    }
}

bool save_to_wav(const std::string &filename,
                 const std::vector<float> &audio_data, int sample_rate) {
  SDL_AudioSpec wav_spec;
  SDL_zero(wav_spec);

  wav_spec.freq = sample_rate;
  wav_spec.format = AUDIO_F32;
  wav_spec.channels = 1;
  wav_spec.samples = 4096;
  wav_spec.callback = nullptr;

  SDL_AudioCVT cvt;
  SDL_zero(cvt);

  if (SDL_BuildAudioCVT(&cvt, wav_spec.format, wav_spec.channels, wav_spec.freq,
                        wav_spec.format, wav_spec.channels,
                        wav_spec.freq) < 0) {
    fprintf(stderr, "Failed to build audio converter: %s\n", SDL_GetError());
    return false;
  }

  cvt.len = audio_data.size() * sizeof(float);
  cvt.buf = (Uint8 *)SDL_malloc(cvt.len * cvt.len_mult);
  memcpy(cvt.buf, audio_data.data(), cvt.len);

  if (SDL_ConvertAudio(&cvt) < 0) {
    fprintf(stderr, "Failed to convert audio: %s\n", SDL_GetError());
    SDL_free(cvt.buf);
    return false;
  }

  SDL_RWops *out = SDL_RWFromFile(filename.c_str(), "wb");
  if (!out) {
    fprintf(stderr, "Failed to open file for writing: %s\n", SDL_GetError());
    SDL_free(cvt.buf);
    return false;
  }

  SDL_WriteLE32(out, 0x46464952);        // "RIFF"
  SDL_WriteLE32(out, cvt.len_cvt + 36);  // Chunk size
  SDL_WriteLE32(out, 0x45564157);        // "WAVE"
  SDL_WriteLE32(out, 0x20746d66);        // "fmt "
  SDL_WriteLE32(out, 16);                // Subchunk size
  SDL_WriteLE16(out, 3);  // Audio format (1 = PCM, 3 = IEEE Float)
  SDL_WriteLE16(out, wav_spec.channels);  // Number of channels
  SDL_WriteLE32(out, wav_spec.freq);      // Sample rate
  SDL_WriteLE32(
      out, wav_spec.freq * wav_spec.channels * sizeof(float));  // Byte rate
  SDL_WriteLE16(out, wav_spec.channels * sizeof(float));        // Block align
  SDL_WriteLE16(out, sizeof(float) * 8);  // Bits per sample
  SDL_WriteLE32(out, 0x61746164);         // "data"
  SDL_WriteLE32(out, cvt.len_cvt);        // Data chunk size
  SDL_RWwrite(out, cvt.buf, cvt.len_cvt, 1);

  SDL_RWclose(out);
  SDL_free(cvt.buf);

  return true;
}
