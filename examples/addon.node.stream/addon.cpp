#include <SDL2/SDL.h>
#include <SDL2/SDL_audio.h>
#include <napi.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common-sdl_2.h"
#include "common.h"
#include "utils.h"
#include "whisper.h"

class WhisperWorker : public Napi::AsyncProgressWorker<std::string> {
public:
    WhisperWorker(Napi::Function& callback, whisper_params& params)
        : Napi::AsyncProgressWorker<std::string>(callback)
        , params(params)
        , shouldStop(false)
    {
    }

    void Stop()
    {
        shouldStop = true;
    }
    ~WhisperWorker()
    {
    }

    bool vad_detection(std::vector<float>& pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold,
        bool wait_for_fade_out, bool verbose)
    {
        // fprintf(stdout, "!!!!!!!!!!!: %f", vad_thold);
        const int n_samples = pcmf32.size();
        const int n_samples_last = (sample_rate * last_ms) / 1000;

        // not enough samples - assume no speec
        if (n_samples_last >= n_samples)
            return false;

        if (freq_thold > 0.0f)
            high_pass_filter(pcmf32, freq_thold, sample_rate);

        float energy_all = 0.0f;
        float energy_last = 0.0f;

        for (int i = 0; i < n_samples; i++) {
            energy_all += fabsf(pcmf32[i]);
            if (i >= n_samples - n_samples_last) {
                energy_last += fabsf(pcmf32[i]);
            }
        }

        energy_all /= n_samples;
        energy_last /= n_samples_last;

        if (verbose) {
            log_debug(__func__, energy_all, energy_last, vad_thold, freq_thold);
        }

        // Подобранное пороговое значение для минимальной энергии
        const float min_energy_threshold = 0.000130f;

        bool speech_detected = wait_for_fade_out
            ? energy_last <= vad_thold * energy_all // Если ждем окончания речи, речь
                                                    // завершается, когда энергия падает
            : energy_last > vad_thold * energy_all && energy_last > min_energy_threshold; // В противном случае
                                                                                          // речь обнаруживается,
                                                                                          // если превышен порог

        return speech_detected;
    }

    void Execute(const ExecutionProgress& progress)
    {
        // Initialize Whisper context
        ctx = init_whisper_context(params, 0, nullptr);

        // audio_async audio(params.hard_ms_th, 0.0040f);
        audio_async audio(params.hard_ms_th);
        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            SetError("Audio initialization failed");
            return;
        }
        audio.resume();

        std::vector<float> pcmf32(n_samples_40s, 0.0f);
        std::vector<float> pcmf32_old;
        std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
        std::vector<float> pcmf32_zcr(n_samples_30s, 0.0f);

        std::vector<whisper_token> prompt_tokens;

        print_processing_info(ctx, params, n_samples_step, n_samples_len, n_samples_keep, use_vad, n_new_line);

        int n_iter = 0;

        const auto t_start = std::chrono::high_resolution_clock::now();

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
        // wparams.debug_mode = true;

        // Определите значения по умолчанию
        const int default_vad_window = 1000;
        const int min_last_ms = 120;
        const int decrement_ms = 100;
        // const int max_ms = 7000;  // Максимальное время для транскрипции, 10
        // секунд
        int vad_window_ms = default_vad_window;
        auto last_sample_time = std::chrono::high_resolution_clock::now();
        int time_since_last = 0;

        bool zcr_detect = false;

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        fprintf(stdout, "Start transcribing...\n");

        while (!shouldStop) {
            // Process audio and transcribe
            if (!use_vad) {
                // if (!use_vad) {
                while (true) {
                    audio.get(params.step_ms, pcmf32_new);
                    if ((int)pcmf32_new.size() > 2 * n_samples_step) {
                        fprintf(stderr,
                            "\n\n%s: WARNING: cannot process audio fast enough, "
                            "dropping audio ...\n\n",
                            __func__);
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
            } else {
                // Stage 1: Waiting
                const auto t_now = std::chrono::high_resolution_clock::now();
                const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start).count();

                if (t_diff < 1000) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                // Stage 2: Voice Detect (VAD)
                // Stage 2.1 get sample for VAD
                const auto vad_sample_ms = time_since_last == 0
                    ? 2000
                    : std::min(2000,
                          static_cast<int>(
                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - last_sample_time)
                                  .count()));

                audio.get(vad_sample_ms, pcmf32_new);

                const bool voice_fade_out_detect = vad_detection(pcmf32_new, WHISPER_SAMPLE_RATE, vad_window_ms, params.vad_thold, params.freq_thold, true, false);

                time_since_last = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - last_sample_time)
                                                       .count());
                const bool over_load = time_since_last > params.hard_ms_th;

                // Stage 2.2 if voice detected get audio
                // fprintf(stdout, "Before VAD: vad_window_ms: %d \n", vad_window_ms);
                // fprintf(stdout, "time_since_last: %d, params.hard_ms_th: %d, overload: %s \n", time_since_last, params.hard_ms_th, over_load ? "ON" : "OFF");
                if (voice_fade_out_detect
                    || (over_load)) {

                    // save_to_wav("vad.wav", pcmf32_new, 16000);

                    int capture_ms = time_since_last;
                    // int capture_ms = std::min(params.hard_ms_th, time_since_last);

                    if (voice_fade_out_detect) {
                        // fprintf(stdout, "[ 👂 ] vad_window_ms: %d, vad_sample_ms: %d", vad_window_ms, vad_sample_ms);
                    }

                    // fprintf(stdout, ".\n");
                    audio.get(std::max(capture_ms, vad_window_ms), pcmf32, true);

                    if (over_load) {
                        auto zcr_detect = vad_detection_windowed_zcr(pcmf32, WHISPER_SAMPLE_RATE, default_vad_window, (params.vad_thold / 10), false);

                        // fprintf(stdout, "[[ 🚨: %d/%d %s]]", capture_ms, params.hard_ms_th, zcr_detect ? "🎧 " : "X");

                        if (!zcr_detect) {

                            last_sample_time = std::chrono::high_resolution_clock::now(); // Сохранить время
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            vad_window_ms = default_vad_window;
                            continue;
                        }

                        // fprintf(stdout, "\n overload ZCR: capture_ms: %d zcr_detect: %s\n", capture_ms, zcr_detect ? "DETECT" : "NOT DETECT");
                    }

                    // if (capture_ms < 1000) {
                    //     fprintf(stdout, "[[ 🚫: %d ]]", capture_ms);
                    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    //     continue;
                    // }

                    // std::cout << std::endl << "[[" << capture_ms << "]]" <<
                    // std::endl;

                    // save_to_wav("output.wav", pcmf32, 16000);

                    last_sample_time = std::chrono::high_resolution_clock::now(); // Сохранить время

                    // Сбросить окно паузы на значение по умолчанию
                    vad_window_ms = default_vad_window;
                } else {
                    // Stage 2.1 change the vad windows if need and continue if no vad detected
                    // time_since_last = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    //     std::chrono::high_resolution_clock::now() - last_sample_time)
                    //                                        .count());

                    // fprintf(stdout, ".\n");

                    audio.get(time_since_last, pcmf32_zcr);

                    zcr_detect = vad_detection_windowed_zcr(pcmf32_zcr, WHISPER_SAMPLE_RATE, time_since_last,
                        (params.vad_thold / 20), false);

                    if (zcr_detect) {
                        fprintf(stdout, "ZCR  time_since_last:%d, vad_window_ms: %d \n", time_since_last, vad_window_ms);
                    }

                    // Уменьшить окно паузы, но не ниже минимального
                    if (zcr_detect && (vad_window_ms > min_last_ms) && time_since_last > params.soft_ms_th) {
                        vad_window_ms = std::max(vad_window_ms - decrement_ms, min_last_ms);
                    }

                    if (time_since_last > params.hard_ms_th) {
                        // fprintf(stdout,
                        //         "⚠️ Warning: Maximum transcription time reached,
                        //         " "some audio may be lost.\n");
                    }
                    // fprintf(stdout,
                    //         "No VAD, elapsed_ms: %d, vad_window_ms: %d,
                    //         params.hard_ms_th: %d, " "soft_ms_th: %d \n",
                    //         elapsed_ms, vad_window_ms,  params.hard_ms_th,
                    //         params.soft_ms_th);

                    // wait for next iteration
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    // time_since_last = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    //     std::chrono::high_resolution_clock::now() - last_sample_time);
                    continue;
                }
            }

            // fprintf(stdout, "%zu NEXT!\n", pcmf32.size());

            // Stage 3: Transcribe

            // Stage 3.1:
            // If data is less than 1000ms, add silence up to 1000ms
            const int required_size = static_cast<int>((1000.0 / 1000.0) * WHISPER_SAMPLE_RATE);

            // Проверяем, нужно ли добавить нули
            if (pcmf32.size() <= required_size) {
                // Вычисляем количество нулей, которые нужно добавить
                fprintf(stdout, "[[ 🚨🚨 %d ms %zu, %d]]", static_cast<int>((pcmf32.size() / static_cast<float>(WHISPER_SAMPLE_RATE)) * 1000.0f), pcmf32.size(), required_size);

                // save_to_wav("output.wav", pcmf32, 16000);
                // exit(0);

                int zeros_to_add = required_size - pcmf32.size();

                fprintf(stdout, "==%d==", zeros_to_add);

                // Создаем вектор с нулями (значениями 0.0020)
                std::vector<float> zeros(zeros_to_add + 100, 0.0020f);

                // Добавляем нули в конец вектора pcmf32
                pcmf32.insert(pcmf32.end(), zeros.begin(), zeros.end());
            }

            auto w_start = std::chrono::high_resolution_clock::now();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stdout, "Error: problem during invocation of 'whisper_full'\n");
                SetError("Failed to process audio: whisper_full");
                return;
            }

            int executed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - w_start).count();
            // // Выводим результат
            // std::cout << "<"
            //           << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - w_start).count()
            //           << ">";

            // Stage 3.2:
            // Send the transcription results to the main thread
            const int n_segments = whisper_full_n_segments(ctx);

            // fprintf(stdout, "Segments - %d \n", n_segments);
            for (int i = 0; i < n_segments; ++i) {
                const char* text = whisper_full_get_segment_text(ctx, i);
                std::string segment_text(text);
                fprintf(stdout, "[%d, %d, %d] #%d: ✅ %s \n", time_since_last, executed, time_since_last - executed, i, text);

                progress.Send(&segment_text, 1);
            }

            // exit(1);

            // fprintf(stdout, "Iteration: %d \n", n_iter);

            ++n_iter;

            // if (!use_vad && (n_iter % n_new_line) == 0) {
            //   pcmf32_old =
            //       std::vector<float>(pcmf32.end() - n_samples_keep,
            //       pcmf32.end());
            // update_prompt_tokens(ctx, prompt_tokens, params.no_context);
            // }
        }

        // Clean up
        audio.pause();

        whisper_print_timings(ctx);
        whisper_free(ctx);
    }

    void OnProgress(const std::string* segment, size_t count) override
    {
        Napi::Env env = Env();
        Napi::HandleScope scope(env);

        Napi::Object obj = Napi::Object::New(env);
        obj.Set("text", Napi::String::New(env, *segment));

        // Include the Stop function in the object sent to the callback
        Napi::Function stopFunction = Napi::Function::New(env, [this](const Napi::CallbackInfo& info) { this->Stop(); });
        obj.Set("stop",
            stopFunction); // Added stop method directly to the callback object

        Callback().Call({ env.Null(), obj });
    }

private:
    whisper_params params;
    whisper_context* ctx;

    bool shouldStop;

    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_len = (1e-3 * params.soft_ms_th) * WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;
    const int n_samples_40s = (1e-3 * 40000.0) * WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0;
    const int n_new_line = !use_vad ? std::max(1, params.soft_ms_th / params.step_ms - 1) : 1;
};

Napi::Value TranscribeAudio(const Napi::CallbackInfo& info)
{
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
    int32_t n_threads = obj.Has("n_threads") ? obj.Get("n_threads").As<Napi::Number>().Int32Value() : 4;
    int32_t step_ms = obj.Has("step_ms") ? obj.Get("step_ms").As<Napi::Number>().Int32Value() : 3000;
    int32_t soft_ms_th = obj.Has("soft_ms_th") ? obj.Get("soft_ms_th").As<Napi::Number>().Int32Value() : 10000; // useles??

    int32_t hard_ms_th = obj.Has("hard_ms_th") ? obj.Get("hard_ms_th").As<Napi::Number>().Int32Value() : 10000; // useles??
    int32_t keep_ms = obj.Has("keep_ms") ? obj.Get("keep_ms").As<Napi::Number>().Int32Value() : 200;
    std::string language = obj.Has("language") ? obj.Get("language").As<Napi::String>().Utf8Value() : "en";
    std::string model = obj.Get("model").As<Napi::String>().Utf8Value();
    int32_t capture_id = obj.Has("capture_id") ? obj.Get("capture_id").As<Napi::Number>().Int32Value() : -1;
    bool translate = obj.Has("translate") ? obj.Get("translate").As<Napi::Boolean>().Value() : false;
    bool use_gpu = obj.Has("use_gpu") ? obj.Get("use_gpu").As<Napi::Boolean>().Value() : true;
    float vad_thold = obj.Has("vad_thold") ? obj.Get("vad_thold").As<Napi::Number>().FloatValue() : 0.6f;

    whisper_params params;
    params.use_gpu = use_gpu;
    params.model = model;
    params.n_threads = n_threads;
    params.step_ms = step_ms;
    params.soft_ms_th = soft_ms_th;
    params.hard_ms_th = hard_ms_th;
    params.keep_ms = keep_ms;
    params.capture_id = capture_id;
    params.translate = translate;
    params.language = language;

    params.keep_ms = std::min(params.keep_ms, params.step_ms);
    params.soft_ms_th = std::max(params.soft_ms_th, params.step_ms);
    params.hard_ms_th = std::max(params.hard_ms_th, params.soft_ms_th);

    params.no_timestamps = false;
    params.no_context = false;
    params.max_tokens = 0;
    params.vad_thold = vad_thold;

    WhisperWorker* worker = new WhisperWorker(callback, params);
    worker->Queue();

    // Создаем функцию обратного вызова для остановки работы
    Napi::Function stopFunction = Napi::Function::New(env, [worker](const Napi::CallbackInfo& info) { worker->Stop(); });

    // Передаем функцию обратного вызова в JavaScript
    Napi::Object n_obj = Napi::Object::New(env);
    n_obj.Set("stop", stopFunction);
    return n_obj;
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(Napi::String::New(env, "transcribeAudio"), Napi::Function::New(env, TranscribeAudio));
    return exports;
}

NODE_API_MODULE(whisper_addon, Init)