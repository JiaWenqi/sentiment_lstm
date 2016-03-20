import numpy
import tensorflow as tf

import math
import model
import load
import time


def main():
  batch_size = 100
  length = 150
  num_class = 2
  learning_rate = 5.0
  epochs = 100000
  voc_size = 100000
  emb_dim = 10

  train, valid, test = load.load_data(n_words=voc_size)
  x, labels = load.prepare_data(train[0], train[1], maxlen=length)
  print('There are %d training cases.' % len(labels))
  test_x, test_labels = load.prepare_data(test[0], test[1], maxlen=length)

  with tf.Graph().as_default():
    x_placeholder = tf.placeholder(tf.int32,
                                   shape=[batch_size, length],
                                   name='label')

    label_placeholder = tf.placeholder(tf.int64,
                                       shape=[batch_size],
                                       name='label')

    lstm = model.LSTM(length=length,
                      batch_size=batch_size,
                      voc_size=voc_size,
                      emb_dim=emb_dim,
                      num_class=num_class)
    inference = lstm.Inference(x_placeholder)
    loss = lstm.Loss(inference, label_placeholder)
    train_op = lstm.Train(loss, learning_rate=learning_rate)
    evaluate = lstm.Evaluate(inference, label_placeholder)

    with tf.Session() as sess:
      print('initializing all variables.')

      init = tf.initialize_all_variables()
      sess.run(init)

      for epoch in range(epochs):
        i = 0
        total_loss = 0
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

        # print(lstm.final_cell.W_o.eval())
        # print(lstm.final_cell.embedding[1:100, :].eval())

        total_test_eval = Evaluate(sess, batch_size, test_x, test_labels,
                                   x_placeholder, label_placeholder, evaluate)

        total_val_eval = Evaluate(sess, batch_size, x, labels, x_placeholder,
                                  label_placeholder, evaluate)

        print(
            'Epoch %d: loss = %.5f ; val_evaluation = %.2f ; test_evaluation = %.2f (%.3f sec)'
            %
            (epoch, total_loss / i, total_val_eval, total_test_eval, duration))


def Evaluate(sess, batch_size, batch_x, batch_label, x_placeholder,
             label_placeholder, evaluate):
  j = 0
  total_eval = 0.0
  while True:
    batch_test_x, batch_test_label = load.NextMiniBatch(batch_x, batch_label, j,
                                                        batch_size)
    j = j + 1
    if batch_test_x is None or batch_test_label is None:
      break

    feed_dict = {x_placeholder: batch_test_x,
                 label_placeholder: batch_test_label}

    evaluate_value = sess.run(evaluate, feed_dict=feed_dict)

    total_eval += evaluate_value

  return total_eval / j


if __name__ == '__main__':
  main()
