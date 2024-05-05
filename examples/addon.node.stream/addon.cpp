#include <napi.h>
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

class WhisperWorker : public Napi::AsyncWorker {
public:
    WhisperWorker(Napi::Function& callback, whisper_params& params)
        : Napi::AsyncWorker(callback), params(params) {}

    ~WhisperWorker() {}

    void Execute() {
        // Initialize Whisper context
        ctx = init_whisper_context(params, 0, nullptr);

        audio_async audio(params.length_ms);
        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            SetError("Audio initialization failed");
            return;
        }
        audio.resume();

        std::vector<float> pcmf32(n_samples_30s, 0.0f);
        std::vector<float> pcmf32_old;
        std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

        std::vector<whisper_token> prompt_tokens;

        print_processing_info(ctx, params, n_samples_step, n_samples_len, n_samples_keep, use_vad, n_new_line);

        int n_iter = 0;
        bool is_running = true;

        const auto t_start = std::chrono::high_resolution_clock::now();

        while (is_running) {
            // Process audio and transcribe
            if (!use_vad) {
                while (true) {
                    audio.get(params.step_ms, pcmf32_new);
                    if ((int)pcmf32_new.size() > 2 * n_samples_step) {
                        fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                        audio.clear();
                        continue;
                    }
                    if ((int)pcmf32_new.size() >= n_samples_step) {
                        audio.clear();
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                const int n_samples_new = pcmf32_new.size();
                const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

                pcmf32.resize(n_samples_new + n_samples_take);

                for (int i = 0; i < n_samples_take; i++) {
                    pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
                }

                memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new * sizeof(float));

                pcmf32_old = pcmf32;
            }
            else {
                const auto t_now = std::chrono::high_resolution_clock::now();
                const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();

                if (t_diff < 2000) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                audio.get(2000, pcmf32_new);

                if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                    audio.get(params.length_ms, pcmf32);
                }
                else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            }

            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress = false;
            wparams.print_special = params.print_special;
            wparams.print_realtime = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate = params.translate;
            wparams.single_segment = !use_vad;
            wparams.max_tokens = params.max_tokens;
            wparams.language = params.language.c_str();
            wparams.n_threads = params.n_threads;

            wparams.audio_ctx = params.audio_ctx;
            wparams.speed_up = params.speed_up;

            wparams.tdrz_enable = params.tinydiarize;

            wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens = params.no_context ? 0 : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                SetError("Failed to process audio");
                return;
            }

            // Store the transcription results
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char* text = whisper_full_get_segment_text(ctx, i);
                std::string segment_text(text);
                transcription_results.push_back(segment_text);
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
                update_prompt_tokens(ctx, prompt_tokens, params.no_context);
            }

            // Check if the audio processing is finished
            if (n_iter >= 1) {
                is_running = false;
            }
        }

        // Clean up
        audio.pause();
        whisper_print_timings(ctx);
        whisper_free(ctx);
    }

    void OnOK() {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        std::vector<napi_value> args;
        args.push_back(env.Null());

        for (const auto& segment : transcription_results) {
            Napi::Object obj = Napi::Object::New(env);
            obj.Set("text", Napi::String::New(env, segment));
            args.push_back(obj);
        }

        Callback().Call(args);
    }

private:
    whisper_params params;
    whisper_context* ctx;
    std::vector<std::string> transcription_results;

    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_len = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0;
    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1;
};

Napi::Value TranscribeAudio(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Wrong number of arguments").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsObject() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Object obj = info[0].ToObject();
    Napi::Function callback = info[1].As<Napi::Function>();

    // Extract parameters from the object
    std::string model = obj.Get("model").As<Napi::String>().Utf8Value();
    int32_t n_threads = obj.Get("n_threads").As<Napi::Number>().Int32Value();
    int32_t step_ms = obj.Get("step_ms").As<Napi::Number>().Int32Value();
    int32_t length_ms = obj.Get("length_ms").As<Napi::Number>().Int32Value();
    int32_t keep_ms = obj.Get("keep_ms").As<Napi::Number>().Int32Value();
    int32_t capture_id = obj.Get("capture_id").As<Napi::Number>().Int32Value();
    bool translate = obj.Get("translate").As<Napi::Boolean>().Value();
    std::string language = obj.Get("language").As<Napi::String>().Utf8Value();

    whisper_params params;
    params.model = model;
    params.n_threads = n_threads;
    params.step_ms = step_ms;
    params.length_ms = length_ms;
    params.keep_ms = keep_ms;
    params.capture_id = capture_id;
    params.translate = translate;
    params.language = language;

    params.keep_ms = std::min(params.keep_ms, params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    params.no_timestamps = false;
    params.no_context = false;
    params.max_tokens = 0;

    WhisperWorker* worker = new WhisperWorker(callback, params);
    worker->Queue();

    return env.Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "transcribeAudio"), Napi::Function::New(env, TranscribeAudio));
    return exports;
}

NODE_API_MODULE(whisper_addon, Init)