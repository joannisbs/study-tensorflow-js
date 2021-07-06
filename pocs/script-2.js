
const tf = require('@tensorflow/tfjs');
// @tensorflow/tfjs-node


(async () => {

  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

  // valores para entrada função 3x+2
  const xs = tf.tensor2d([0, 1, 2, 3, 4, 5], [6, 1]);

  // valores saidas y
  const ys = tf.tensor2d([2, 5, 8, 11, 14, 17], [6, 1]);

  await model.fit(xs, ys, {
    epochs: 1000, // change epochs
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    }
  })

  // prevendo o resultado x = 10, 30 + 2 = 32
  const result = model.predict(tf.tensor2d([10], [1,1]))
  console.log(result)

  // convertendo o tensor para numero
  const dataArray = result.arraySync();
  console.log(dataArray)

  // 32.87376403808594 com 100 epochs
  // 32.112335205078125 com 500 epochs
  // 32.00002670288086 com 10.000 epochs
})()
