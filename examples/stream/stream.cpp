#include "common-sdl.h"
#include "common.h"
#include "whisper.h"
#include "params.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

struct whisper_context* init_whisper_context(const whisper_params& params, int argc, char** argv) {
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
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