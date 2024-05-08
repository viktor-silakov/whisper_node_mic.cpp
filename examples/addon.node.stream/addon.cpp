#include "common-sdl.h"
#include "common.h"
#include "utils.h"
#include "whisper.h"
#include <napi.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

class WhisperWorker : public Napi::AsyncProgressWorker<std::string> {
public:
  WhisperWorker(Napi::Function &callback, whisper_params &params)
      : Napi::AsyncProgressWorker<std::string>(callback), params(params),
        shouldStop(false) {}

  void Stop() { shouldStop = true; }
  ~WhisperWorker() {}

  void log_debug(const char* func, float energy_all, float energy_last, float vad_thold, float freq_thold) {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    long long milliseconds = value.count();

    auto now_time = std::chrono::system_clock::to_time_t(now);
    char timestamp[24];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
    
    fprintf(stderr, "[%s.%03lld] %s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n",
            timestamp, milliseconds % 1000, func, energy_all, energy_last, vad_thold, freq_thold);
  }

  bool vad_detection(std::vector<float> &pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool wait_for_fade_out, bool verbose) {
    const int n_samples = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    // not enough samples - assume no speec
    if (n_samples_last >= n_samples) return false;

    if (freq_thold > 0.0f) high_pass_filter(pcmf32, freq_thold, sample_rate);

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
        ? energy_last <= vad_thold * energy_all // Если ждем окончания речи, речь завершается, когда энергия падает
        : energy_last > vad_thold * energy_all && energy_last > min_energy_threshold; // В противном случае речь обнаруживается, если превышен порог

    return speech_detected;
  }

  void Execute(const ExecutionProgress &progress) {
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

    print_processing_info(ctx, params, n_samples_step, n_samples_len,
                          n_samples_keep, use_vad, n_new_line);

    int n_iter = 0;

    const auto t_start = std::chrono::high_resolution_clock::now();

    whisper_full_params wparams =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

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

    wparams.temperature_inc =
        params.no_fallback ? 0.0f : wparams.temperature_inc;

    wparams.prompt_tokens = params.no_context ? nullptr : prompt_tokens.data();
    wparams.prompt_n_tokens = params.no_context ? 0 : prompt_tokens.size();

    while (!shouldStop) {
      // Process audio and transcribe
      if (!use_vad) {
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
        const int n_samples_take = std::min(
            (int)pcmf32_old.size(),
            std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        pcmf32.resize(n_samples_new + n_samples_take);

        for (int i = 0; i < n_samples_take; i++) {
          pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
        }

        memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(),
               n_samples_new * sizeof(float));

        pcmf32_old = pcmf32;
      } else {
        // Stage 1: Waiting
        const auto t_now = std::chrono::high_resolution_clock::now();
        const auto t_diff =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_now -
                                                                  t_start)
                .count();

        if (t_diff < 2000) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
        // Stage 2: Voice Detect (VAD)
        // Stage 2.1 get sample for VAD
        audio.get(1100, pcmf32_new);


        // Stage 2.2 if voice detected get length_ms audio
        if (vad_detection(pcmf32_new, WHISPER_SAMPLE_RATE, 1000,params.vad_thold, params.freq_thold, true,  true)) {
          
          fprintf(stdout, "VAD!\n");

          audio.get(params.length_ms, pcmf32);

          // fprintf(stdout, "VAD AFTER GET!\n");

        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
      }
      // fprintf(stdout, "%zu NEXT!\n", pcmf32.size());

      // Stage 3: Transcribe
      // Stage 3.1: 

      if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        fprintf(stdout, "Error: problem during invocation of 'whisper_full'\n");
        SetError("Failed to process audio: whisper_full");
        return;
      }

      // std::this_thread::sleep_for(std::chrono::milliseconds(1500));

      // Stage 3.2: 
      // Send the transcription results to the main thread
      const int n_segments = whisper_full_n_segments(ctx);

      // fprintf(stdout, "Segments - %d \n", n_segments);
      for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        std::string segment_text(text);
        fprintf(stdout, "TEXT #%d: %s \n", i, text);

        progress.Send(&segment_text, 1);
      }

      // fprintf(stdout, "Iteration: %d \n", n_iter);

      ++n_iter;

      // if (!use_vad && (n_iter % n_new_line) == 0) {
      //   pcmf32_old =
      //       std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
      //   update_prompt_tokens(ctx, prompt_tokens, params.no_context);
      // }

    }

    // Clean up
    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);
  }

  void OnProgress(const std::string *segment, size_t count) override {
    Napi::Env env = Env();
    Napi::HandleScope scope(env);

    Napi::Object obj = Napi::Object::New(env);
    obj.Set("text", Napi::String::New(env, *segment));

    // Include the Stop function in the object sent to the callback
    Napi::Function stopFunction = Napi::Function::New(
        env, [this](const Napi::CallbackInfo &info) { this->Stop(); });
    obj.Set(
        "stop",
        stopFunction); // Added stop method directly to the callback object

    Callback().Call({env.Null(), obj});
  }

private:
  whisper_params params;
  whisper_context *ctx;

  bool shouldStop;

  const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
  const int n_samples_len = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
  const int n_samples_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;
  const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

  const bool use_vad = n_samples_step <= 0;
  const int n_new_line =
      !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1;
};

Napi::Value TranscribeAudio(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Wrong number of arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[0].IsObject() || !info[1].IsFunction()) {
    Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Object obj = info[0].ToObject();
  Napi::Function callback = info[1].As<Napi::Function>();

  // Extract parameters from the object
  int32_t n_threads = obj.Has("n_threads")
                          ? obj.Get("n_threads").As<Napi::Number>().Int32Value()
                          : 4;
  int32_t step_ms = obj.Has("step_ms")
                        ? obj.Get("step_ms").As<Napi::Number>().Int32Value()
                        : 3000;
  int32_t length_ms = obj.Has("length_ms")
                          ? obj.Get("length_ms").As<Napi::Number>().Int32Value()
                          : 10000; // useles??
  int32_t keep_ms = obj.Has("keep_ms")
                        ? obj.Get("keep_ms").As<Napi::Number>().Int32Value()
                        : 200;
  std::string language =
      obj.Has("language") ? obj.Get("language").As<Napi::String>().Utf8Value()
                          : "en";
  std::string model = obj.Get("model").As<Napi::String>().Utf8Value();
  int32_t capture_id =
      obj.Has("capture_id")
          ? obj.Get("capture_id").As<Napi::Number>().Int32Value()
          : -1;
  bool translate = obj.Has("translate")
                       ? obj.Get("translate").As<Napi::Boolean>().Value()
                       : false;
  bool use_gpu = obj.Has("use_gpu")
                     ? obj.Get("use_gpu").As<Napi::Boolean>().Value()
                     : true;

  whisper_params params;
  params.use_gpu = use_gpu;
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

  WhisperWorker *worker = new WhisperWorker(callback, params);
  worker->Queue();

  // Создаем функцию обратного вызова для остановки работы
  Napi::Function stopFunction = Napi::Function::New(
      env, [worker](const Napi::CallbackInfo &info) { worker->Stop(); });

  // Передаем функцию обратного вызова в JavaScript
  Napi::Object n_obj = Napi::Object::New(env);
  n_obj.Set("stop", stopFunction);
  return n_obj;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "transcribeAudio"),
              Napi::Function::New(env, TranscribeAudio));
  return exports;
}

NODE_API_MODULE(whisper_addon, Init)