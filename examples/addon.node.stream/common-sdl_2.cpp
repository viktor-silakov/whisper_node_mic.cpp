#include "common-sdl_2.h"

#include <cstdio>
#include <iomanip>
#include <iostream>
// #include "common-sdl.h"

audio_async::audio_async(int len_ms, bool ignore_silence) {
  m_len_ms = len_ms;
  m_ignore_silence = ignore_silence;
  m_running = false;
  // m_audio.resize(100, 0.0055f); // Инициализируем буфер одним нулевым
  // значением
}

audio_async::~audio_async() {
  if (m_dev_id_in) {
    SDL_CloseAudioDevice(m_dev_id_in);
  }
}

bool audio_async::init(int capture_id, int sample_rate) {
  SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

  if (SDL_Init(SDL_INIT_AUDIO) < 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n",
                 SDL_GetError());
    return false;
  }

  SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium",
                          SDL_HINT_OVERRIDE);

  {
    int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
    fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
    for (int i = 0; i < nDevices; i++) {
      fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i,
              SDL_GetAudioDeviceName(i, SDL_TRUE));
    }
  }

  SDL_AudioSpec capture_spec_requested;
  SDL_AudioSpec capture_spec_obtained;

  SDL_zero(capture_spec_requested);
  SDL_zero(capture_spec_obtained);

  capture_spec_requested.freq = sample_rate;
  capture_spec_requested.format = AUDIO_F32;
  capture_spec_requested.channels = 1;
  capture_spec_requested.samples = 1024;

  if (m_ignore_silence) {
    capture_spec_requested.callback = [](void* userdata, uint8_t* stream,
                                         int len) {
      audio_async* audio = (audio_async*)userdata;
      audio->callback_ignore_silence(stream, len);
    };
  } else {
    capture_spec_requested.callback = [](void* userdata, uint8_t* stream,
                                         int len) {
      audio_async* audio = (audio_async*)userdata;
      audio->callback(stream, len);
    };
  }

  //   capture_spec_requested.callback = [](void* userdata, uint8_t* stream,
  //                                        int len) {
  //     audio_async* audio = (audio_async*)userdata;
  //     audio->callback(stream, len);
  //   };
  capture_spec_requested.userdata = this;

  if (capture_id >= 0) {
    fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n",
            __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
    m_dev_id_in = SDL_OpenAudioDevice(
        SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE,
        &capture_spec_requested, &capture_spec_obtained, 0);
  } else {
    fprintf(stderr, "%s: attempt to open default capture device ...\n",
            __func__);
    m_dev_id_in = SDL_OpenAudioDevice(
        nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
  }

  if (!m_dev_id_in) {
    fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n",
            __func__, SDL_GetError());
    m_dev_id_in = 0;

    return false;
  } else {
    fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n",
            __func__, m_dev_id_in);
    fprintf(stderr, "%s:     - sample rate:       %d\n", __func__,
            capture_spec_obtained.freq);
    fprintf(stderr, "%s:     - format:            %d (required: %d)\n",
            __func__, capture_spec_obtained.format,
            capture_spec_requested.format);
    fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",
            __func__, capture_spec_obtained.channels,
            capture_spec_requested.channels);
    fprintf(stderr, "%s:     - samples per frame: %d\n", __func__,
            capture_spec_obtained.samples);
  }

  m_sample_rate = capture_spec_obtained.freq;

  m_audio.resize((m_sample_rate * m_len_ms) / 1000);

  return true;
}

bool audio_async::resume() {
  if (!m_dev_id_in) {
    fprintf(stderr, "%s: no audio device to resume!\n", __func__);
    return false;
  }

  if (m_running) {
    fprintf(stderr, "%s: already running!\n", __func__);
    return false;
  }

  SDL_PauseAudioDevice(m_dev_id_in, 0);

  m_running = true;

  return true;
}

bool audio_async::pause() {
  if (!m_dev_id_in) {
    fprintf(stderr, "%s: no audio device to pause!\n", __func__);
    return false;
  }

  if (!m_running) {
    fprintf(stderr, "%s: already paused!\n", __func__);
    return false;
  }

  SDL_PauseAudioDevice(m_dev_id_in, 1);

  m_running = false;

  return true;
}

bool audio_async::clear() {
  if (!m_dev_id_in) {
    fprintf(stderr, "%s: no audio device to clear!\n", __func__);
    return false;
  }

  if (!m_running) {
    fprintf(stderr, "%s: not running!\n", __func__);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_audio_pos = 0;
    m_audio_len = 0;
  }

  return true;
}

void audio_async::print_energy() {
  if (!m_running) {
    return;
  }

  std::vector<float> buffer;
  get(0, buffer);

  float energy = 0.0f;
  for (const auto& sample : buffer) {
    energy += sample * sample;
  }
  energy = std::sqrt(energy / buffer.size());

  int stars = static_cast<int>(energy / 0.0050 * 15);
  std::cout << "Sound energy: " << std::fixed << std::setprecision(4) << energy
            << " [";
  for (int i = 0; i < stars; ++i) {
    std::cout << "*";
  }
  for (int i = stars; i < 15; ++i) {
    std::cout << " ";
  }
  std::cout << "]" << std::endl;
}

// callback to be called by SDL
// void audio_async::callback(uint8_t* stream, int len) {

void audio_async::callback(uint8_t* stream, int len) {
  if (!m_running) {
    return;
  }

  size_t n_samples = len / sizeof(float);

  if (n_samples > m_audio.size()) {
    n_samples = m_audio.size();

    stream += (len - (n_samples * sizeof(float)));
  }

  // fprintf(stderr, "%s: %zu samples, pos %zu, len %zu\n", __func__, n_samples,
  // m_audio_pos, m_audio_len);

  {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_audio_pos + n_samples > m_audio.size()) {
      const size_t n0 = m_audio.size() - m_audio_pos;

      memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
      memcpy(&m_audio[0], stream + n0 * sizeof(float),
             (n_samples - n0) * sizeof(float));

      m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
      m_audio_len = m_audio.size();
    } else {
      memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));

      m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
      m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }
  }
  // print_energy();
}

int audio_async::get_total_silence_ms() {
  int result = m_total_silence_ms;
  m_total_silence_ms = 0;
  return result;
}

void audio_async::callback_ignore_silence(uint8_t* stream, int len) {
  if (!m_running) {
    return;
  }

  size_t n_samples = len / sizeof(float);

  if (n_samples > m_audio.size()) {
    n_samples = m_audio.size();

    stream += (len - (n_samples * sizeof(float)));
  }

  std::vector<float> temp_buffer(n_samples);
  memcpy(temp_buffer.data(), stream, n_samples * sizeof(float));

  float energy = 0.0f;
  for (const auto& sample : temp_buffer) {
    energy += sample * sample;
  }
  energy = std::sqrt(energy / temp_buffer.size());

  // static bool is_filled = false;  // Локальная переменная is_filled

  if (energy >= 0.0030) {
    // Если энергия выше порога, записываем семплы в буфер
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_audio_pos + n_samples > m_audio.size()) {
      const size_t n0 = m_audio.size() - m_audio_pos;

      memcpy(&m_audio[m_audio_pos], temp_buffer.data(), n0 * sizeof(float));
      memcpy(&m_audio[0], temp_buffer.data() + n0,
             (n_samples - n0) * sizeof(float));

      m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
      m_audio_len = m_audio.size();
    } else {
      memcpy(&m_audio[m_audio_pos], temp_buffer.data(),
             n_samples * sizeof(float));

      m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
      m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
    }

    m_current_silence_ms = 0;  // Сбрасываем счетчик тишины
    m_is_filled = false;  // Сбрасываем флаг заполнения буфера
  } else {
    // Если энергия ниже порога, увеличиваем счетчик тишины
    int silence_ms = (n_samples * 1000) / m_sample_rate;
    m_current_silence_ms += silence_ms;
    m_total_silence_ms += silence_ms;

    if (m_current_silence_ms >= 500 && !m_is_filled) {
      fprintf(stdout, "🍎\n");
      // Если пауза тишины длится 500 мс или более и буфер еще не был заполнен,
      // заполняем буфер значениями 0.0020f
      std::lock_guard<std::mutex> lock(m_mutex);

      size_t fill_samples = (m_sample_rate * 500) / 1000;
      if (fill_samples > m_audio.size()) {
        fill_samples = m_audio.size();
      }

      const float filler = 0.0020f;

      if (m_audio_pos + fill_samples > m_audio.size()) {
        const size_t n0 = m_audio.size() - m_audio_pos;
        std::fill(&m_audio[m_audio_pos], &m_audio[m_audio_pos] + n0, filler);
        std::fill(&m_audio[0], &m_audio[fill_samples - n0], filler);

        m_audio_pos = (m_audio_pos + fill_samples) % m_audio.size();
        m_audio_len = m_audio.size();
      } else {
        std::fill(&m_audio[m_audio_pos], &m_audio[m_audio_pos] + fill_samples,
                  filler);

        m_audio_pos = (m_audio_pos + fill_samples) % m_audio.size();
        m_audio_len = std::min(m_audio_len + fill_samples, m_audio.size());
      }

      m_is_filled = true;  // Устанавливаем флаг заполнения буфера
      m_current_silence_ms = 0;
    }
  }
}

int audio_async::get(int ms, std::vector<float>& result, bool return_silence) {
  if (!m_dev_id_in) {
    fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
    return 0;
  }

  if (!m_running) {
    fprintf(stderr, "%s: not running!\n", __func__);
    return 0;
  }

  result.clear();

  {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (ms <= 0) {
      ms = m_len_ms;
    }

    size_t n_samples = (m_sample_rate * ms) / 1000;
    if (n_samples > m_audio_len) {
      n_samples = m_audio_len;
    }

    result.resize(n_samples);

    int s0 = m_audio_pos - n_samples;
    if (s0 < 0) {
      s0 += m_audio.size();
    }

    if (s0 + n_samples > m_audio.size()) {
      const size_t n0 = m_audio.size() - s0;

      memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
      memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
    } else {
      memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
    }
  }
  if (return_silence) {
    int total_silence_ms = get_total_silence_ms();
    m_total_silence_ms = 0;
    // fprintf(stdout, "tmp: %d\n", total_silence_ms);
    return total_silence_ms;
  }

  return 0;

  // print_energy();
}

bool sdl_poll_events() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
      case SDL_QUIT: {
        return false;
      } break;
      default:
        break;
    }
  }

  return true;
}