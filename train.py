import numpy
import tensorflow as tf

import math
import model
import load
import time


def main():
  batch_size = 100
  length = 100
  input_dim = 1
  output_dim = 1
  learning_rate = 0.1
  epochs = 100000

  train, valid, test = load.load_data()
  x, labels = load.prepare_data(train[0], train[1], maxlen=length)
  test_x, test_labels = load.prepare_data(test[0], test[1], maxlen=length)

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
    evaluate = lstm.Evaluate(inference, label_placeholder)

    with tf.Session() as sess:
      print('initializing al variables.')

      init = tf.initialize_all_variables()
      sess.run(init)

      for epoch in range(epochs):
        i = 0
        total_loss = 0
        j = 0
        total_eval = 0
        while True:
          start_time = time.time()
          batch_x, batch_label = load.NextMiniBatch(x, labels, i, batch_size)
          i = i + 1
          if batch_x is None or batch_label is None:
            break

          feed_dict = {x_placeholder: batch_x, label_placeholder: batch_label}

          _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

          total_loss += loss_value

          duration = time.time() - start_time

        while True:
          batch_test_x, batch_test_label = load.NextMiniBatch(
              test_x, test_labels, j, batch_size)
          j = j + 1
          if batch_test_x is None or batch_test_label is None:
            break

          feed_dict = {x_placeholder: batch_test_x,
                       label_placeholder: batch_test_label}

          evaluate_value = sess.run(evaluate, feed_dict=feed_dict)

          total_eval += evaluate_value

        print('Epoch %d: loss = %.2f : evaluation = %.2f (%.3f sec)' %
              (epoch, total_loss / i, total_eval / j, duration))


if __name__ == '__main__':
  main()
