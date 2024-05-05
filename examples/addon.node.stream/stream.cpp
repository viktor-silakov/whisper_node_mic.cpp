#include "common-sdl.h"
#include "common.h"
#include "whisper.h"
#include "params.h"
#include "utils.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0;
    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1;

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }
    audio.resume();

    whisper_context* ctx = init_whisper_context(params, argc, argv);

    std::vector<float> pcmf32(n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    print_processing_info(ctx, params, n_samples_step, n_samples_len, n_samples_keep, use_vad, n_new_line);

    int n_iter = 0;
    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        if (!open_output_file(fout, params.fname_out)) {
            return 1;
        }
    }

    printf("[Start speaking]\n");
    fflush(stdout);

    const auto t_start = std::chrono::high_resolution_clock::now();

    while (is_running) {
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        if (!use_vad) {
            while (true) {
                audio.get(params.step_ms, pcmf32_new);
                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }
                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
        }

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.print_progress   = false;
        wparams.print_special    = params.print_special;
        wparams.print_realtime   = false;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.translate        = params.translate;
        wparams.single_segment   = !use_vad;
        wparams.max_tokens       = params.max_tokens;
        wparams.language         = params.language.c_str();
        wparams.n_threads        = params.n_threads;

        wparams.audio_ctx        = params.audio_ctx;
        wparams.speed_up         = params.speed_up;

        wparams.tdrz_enable      = params.tinydiarize;

        wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

        wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
        wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
            return 6;
        }

        print_whisper_result(ctx, params, use_vad, n_iter, t_start, fout);

        ++n_iter;

        if (!use_vad && (n_iter % n_new_line) == 0) {
            printf("\n");
            pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
            update_prompt_tokens(ctx, prompt_tokens, params.no_context);
        }

        fflush(stdout);
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}