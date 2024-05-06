// stream.js

// const WhisperWrapper = require('../../build/Release/addon.node.stream');

const whisperAddon = require('../../build/Release/addon.node.stream');

// const whisperAddon = require('./build/Release/whisper_addon.node');

const params = {
  model: '../../models/ggml-base.en.bin',
  n_threads: 8,
  step_ms: 500,
  length_ms: 5000,
  keep_ms: 500,
  // capture_id: -1,
  // translate: false,
  // language: 'en',
  use_gpu: false
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

const worker  = whisperAddon.transcribeAudio(params, (err, segment) => {
  console.log('👹')
  if (err) {
    console.error('Error:', err);
    return;
  }

  if (segment) {
    console.log('Transcription:', segment.text);
  } else {
    console.log('Transcription finished');
  }
});

console.log(worker);

// Некоторое время спустя, когда нужно остановить транскрибирование
setTimeout(() => {
    // Остановка транскрибирования
    worker.stop();
}, 5000); // Например, остановить через 5 секунд

