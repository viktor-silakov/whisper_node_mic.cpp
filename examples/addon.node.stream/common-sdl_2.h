#pragma once

#include <SDL.h>
#include <SDL_audio.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

//
// SDL Audio capture
//

class audio_async {
 public:
  // audio_async(int len_ms);
  audio_async(int len_ms, bool ignore_silence = true);
  ~audio_async();

  bool init(int capture_id, int sample_rate);

  // start capturing audio via the provided SDL callback
  // keep last len_ms seconds of audio in a circular buffer
  bool resume();
  bool pause();
  bool clear();

  // callback to be called by SDL
  void callback(uint8_t* stream, int len);
  void print_energy();
  int get_total_silence_ms();
  void callback_ignore_silence(uint8_t* stream, int len);

  // get audio data from the circular buffer
  int get(int ms, std::vector<float>& audio, bool return_silence = false);

 private:
  SDL_AudioDeviceID m_dev_id_in = 0;

  int m_len_ms = 0;
  int m_sample_rate = 0;
  bool m_ignore_silence = false;
  int m_total_silence_ms = 0;

  std::atomic_bool m_running;
  std::mutex m_mutex;

  std::vector<float> m_audio;
  size_t m_audio_pos = 0;
  size_t m_audio_len = 0;
};

// Return false if need to quit
bool sdl_poll_events();
