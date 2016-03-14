import numpy
import tensorflow as tf

import model
import load
import time


def main():
  batch_size = 10
  length = 50
  input_dim = 1
  output_dim = 1
  learning_rate = 0.001
  epochs = 100

  train, valid, test = load.load_data()
  x, labels = load.prepare_data(train[0], train[1], maxlen=length)

  with tf.Graph().as_default():
    x_placeholder = tf.placeholder(tf.float32,
                                   shape=[batch_size, length * input_dim],
                                   name='label')

    label_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size, output_dim],
                                       name='label')

    lstm = model.LSTM(length=length,
                      batch_size=batch_size,
                      input_dim=input_dim,
                      output_dim=output_dim)
    inference = lstm.Inference(x_placeholder)
    loss = lstm.Loss(inference, label_placeholder)
    train_op = lstm.Train(loss, learning_rate=learning_rate)

    with tf.Session() as sess:
      print('initializing al variables.')

      init = tf.initialize_all_variables()
      sess.run(init)

      for epoch in range(epochs):
        i = 0
        while True:
          start_time = time.time()
          batch_x, batch_label = load.NextMiniBatch(x, labels, i, batch_size)
          i = i + 1
          if batch_x is None or batch_label is None:
            break

          feed_dict = {x_placeholder: batch_x, label_placeholder: batch_label}

          _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
          duration = time.time() - start_time

          if i % 1 == 0:
            print('Epoch %d Batch %d: loss = %.2f (%.3f sec)' %
                  (epoch, i, loss_value, duration))


if __name__ == '__main__':
  main()
