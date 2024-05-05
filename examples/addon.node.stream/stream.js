// stream.js

// const WhisperWrapper = require('../../build/Release/addon.node.stream');

const whisperAddon = require('../../build/Release/addon.node.stream');

console.log(whisperAddon)

const params = {
  model: '../../models/ggml-base.en.bin',
  n_threads: 4,
  step_ms: 3000,
  length_ms: 10000,
  keep_ms: 200,
  capture_id: -1,
  translate: false,
  language: 'en',
};

console.log('Starting audio transcription...');

const result = whisperAddon.transcribeAudio(params);

console.log('Transcription result:');
console.log(result.text);