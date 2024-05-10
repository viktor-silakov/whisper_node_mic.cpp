// stream.js
const whisperAddon = require('../../build/Release/addon.node.stream');


const params = {
  model: '../../models/ggml-base.en.bin',
  n_threads: 8, // number of threads
  // step_ms: 500,
  step_ms: 0, // audio step size
              // if step_ms is null voise autodetect (vad) activates (!) low process utilization!
  length_ms: 3000, // audio length
  keep_ms: 500, // audio to keep from previous step
  // capture_id: -1,
  capture_id: 1, // input device id (-1 for auto)
  // translate: false,
  // language: 'en',
  // use_gpu: false
};

const worker = whisperAddon.transcribeAudio(params, (err, data) => {
  if (err) {
    console.error('Error:', err);
    return;
  }

  if (data) {
    console.log('Transcription: ✅', data.text);
    // console.dir(data)
    if (data.text.toString().toLowerCase().includes('stop transcription')) {
      console.log('💀 Stop detected!');
      data.stop()
    }
  } else {
    console.log('Transcription finished');
  }
});

process.on('SIGINT', () => {
  console.log('\nReceived SIGINT. 👋 Bye!');
  worker.stop();
  process.exit(1);
});


// console.log(worker);

// Некоторое время спустя, когда нужно остановить транскрибирование
setTimeout(() => {
  // Остановка транскрибирования
  // worker.stop();
}, 5000); // Например, остановить через 5 секунд

