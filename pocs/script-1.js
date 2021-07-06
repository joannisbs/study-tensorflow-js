const tf = require('@tensorflow/tfjs');


(async () => {
  // Optional Load the binding:
  // Use '@tensorflow/tfjs-node-gpu' if running with GPU.
  // require('@tensorflow/tfjs-node');

  // Train a simple model:
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [1]}));
  model.add(tf.layers.dense({units: 1, activation: 'linear'}));
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

  const xs = tf.randomNormal([100, 1]);
  const ys = tf.randomNormal([100, 1]);

  await model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    }
  })

  const result = model.predict(tf.tensor2d([20], [1,1]))
  console.log(result)


})()


