// stream.js

// const WhisperWrapper = require('../../build/Release/addon.node.stream');

const whisperAddon = require('../../build/Release/addon.node.stream');

// const whisperAddon = require('./build/Release/whisper_addon.node');

const params = {
  model: '../../models/ggml-base.en.bin',
  n_threads: 4,
  step_ms: 3000,
  length_ms: 30000,
  keep_ms: 200,
  capture_id: -1,
  translate: false,
  language: 'en'
};

// const whisperAddon = require('./build/Release/whisper_addon.node');

// const params = {
//   model: 'path/to/model',
//   n_threads: 4,
//   step_ms: 3000,
//   length_ms: 10000,
//   keep_ms: 200,
//   capture_id: -1,
//   translate: false,
//   language: 'en'
// };

whisperAddon.transcribeAudio(params, (err, ...segments) => {
  console.log('🐷');
  if (err) {
    console.error('Error:', err);
    return;
  }

  segments.forEach(segment => {
    if (segment) {
      console.log('Transcription:', segment.text);
    }
  });

  if (segments.length === 0) {
    console.log('Transcription finished');
  }
});