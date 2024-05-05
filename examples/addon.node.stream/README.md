---
runme:
  id: 01HX40D7Y35FCSRE5R8Z2RMVP4
  version: v3
---

# addon

This is an addon demo that can **perform whisper model reasoning in `node` and `electron` environments**, based on [cmake-js](https://github.com/cmake-js/cmake-js).
It can be used as a reference for using the whisper.cpp project in other node projects.

## Install

```shell {"id":"01HX40D7Y1VM75W12HGW6S8K93"}
cd example/addon.node.stream && npm install
```

## Compile

Make sure it is in the project root directory and compiled with make-js.

```shell {"id":"01HX40D7Y1VM75W12HGZ2ZWKTM"}
# npx cmake-js compile -T addon.node.stream -B Release
npm combile
```

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
