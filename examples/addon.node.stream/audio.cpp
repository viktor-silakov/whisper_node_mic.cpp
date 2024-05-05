#include "audio.h"
#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <thread>

bool init_audio(audio_async& audio, int capture_id, int sample_rate) {
    if (!audio.init(capture_id, sample_rate)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return false;
    }
    audio.resume();
    return true;
}

void create_wav_writer(wav_writer& wavWriter, const std::string& filename, int sample_rate) {
    wavWriter.open(filename, sample_rate, 16, 1);
}

std::string get_current_datetime() {
    std::time_t now = std::time(nullptr);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", std::localtime(&now));
    return std::string(buffer);
}

bool process_audio_non_vad(audio_async& audio, std::vector<float>& pcmf32_new, int n_samples_step, int step_ms) {
    while (true) {
        audio.get(step_ms, pcmf32_new);
        if ((int) pcmf32_new.size() > 2*n_samples_step) {
            fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
            audio.clear();
            continue;
        }
        if ((int) pcmf32_new.size() >= n_samples_step) {
            audio.clear();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

bool process_audio_vad(audio_async& audio, std::vector<float>& pcmf32_new, std::vector<float>& pcmf32, int sample_rate, int length_ms, float vad_thold, float freq_thold) {
    static auto t_last = std::chrono::high_resolution_clock::now();
    const auto t_now  = std::chrono::high_resolution_clock::now();
    const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
    if (t_diff < 2000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return false;
    }
    audio.get(2000, pcmf32_new);
    if (::vad_simple(pcmf32_new, sample_rate, 1000, vad_thold, freq_thold, false)) {
        audio.get(length_ms, pcmf32);
        t_last = t_now;
        return true;
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return false;
    }
}