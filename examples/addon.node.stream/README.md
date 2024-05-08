# Node.js Whisper.cpp Real-time Microphone Stream Transcription Addon
This addon demonstrates the capability of performing real-time speech transcription using the Whisper model in Node.js environments. It captures audio from the microphone, processes it in real-time, and provides transcription results as the audio is being spoken. The addon supports streaming audio input and offers various configuration options to customize the transcription process.

> **Note:** This project is currently in a demo state and is not actively maintained or developed. It serves as a reference implementation and a starting point for integrating the Whisper model into Node.js applications.

## Install

```shell {"id":"01HX40D7Y1VM75W12HGW6S8K93"}
cd example/addon.node.stream && npm install
```

## Compile

Make sure it is in the project root directory and compiled with make-js.

```shell {"id":"01HX40D7Y1VM75W12HGZ2ZWKTM"}
# npx cmake-js compile -T addon.node.stream -B Release
npm compile
```
Rebuild: 
```shell
# rebuild
npm rebuild
```

## Usage Example 

```js
// stream.js
const whisperAddon = require('../../build/Release/addon.node.stream');


const params = {
  model: '../../models/ggml-base.en.bin',
  n_threads: 8,
  step_ms: 500,
  step_ms: 0, // if step_ms is null voise autodetect (vad) activates (!) low process utilization!
  length_ms: 3000, // audio length
  keep_ms: 500, // audio to keep from previous step
  // capture_id: -1,
  capture_id: 1, // input device id (-1 for auto)
  translate: false,
  language: 'en',
  use_gpu: false
};

const worker = whisperAddon.transcribeAudio(params, (err, data) => {
  if (err) {
    console.error('Error:', err);
    return;
  }

  if (data) {
    console.log('Transcription:', data.text);
    if (data.text.toString().toLowerCase().includes('stop transcription')) {
      console.log('💀 Stop phrase detected!');
      data.stop()
    }
  } else {
    console.log('Transcription finished');
  }
});
```


-----

### From old readme file

For Electron addon and cmake-js options, you can see [cmake-js](https://github.com/cmake-js/cmake-js) and make very few configuration changes.

Such as appointing special cmake path:

```shell {"id":"01HX40D7Y29CNJR5V915PP2XZT"}
npx cmake-js compile -c 'xxx/cmake' -T addon.node.stream -B Release
```

```shell {"id":"01HX40D7Y35FCSRE5R8YQB607Q"}
cd examples/addon.node.stream

# node index.js --language='language' --model='model-path' --fname_inp='file-path'
npm start
```

Because this is a simple Demo, only the above parameters are set in the node environment.

Other parameters can also be specified in the node environment.
