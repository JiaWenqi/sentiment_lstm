import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle

import os
import math
import model
import load
import time
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict', help='Prediction mode', action='store_true')
  args = parser.parse_args()

  batch_size = 100
  length = 150
  num_class = 2
  learning_rate = 4.0
  epochs = 100000
  voc_size = 100000
  emb_dim = 300
  pretrained_emb_path = '../data/imdb.emb.pkl'
  checkpoint_path = '../data/lstm.checkpoint'

  train, valid, test = load.load_data(n_words=voc_size)
  x, labels = load.prepare_data(train[0], train[1], maxlen=length)
  print('There are %d training cases.' % len(labels))
  valid_x, valid_labels = load.prepare_data(valid[0], valid[1], maxlen=length)
  test_x, test_labels = load.prepare_data(test[0], test[1], maxlen=length)

  if not os.path.isfile(pretrained_emb_path):
    print(
        'pretrained embedding does not exist, fall back to randomly initialized embedding.')
    pretrained_emb = None
  else:
    with open(pretrained_emb_path, 'r') as f:
      pretrained_emb = pickle.load(f)
    print('pretrained embedding loaded.')

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
                      num_class=num_class,
                      pretrained_emb=pretrained_emb)
    inference = lstm.Inference(x_placeholder)
    loss = lstm.Loss(inference, label_placeholder)
    train_op = lstm.Train(loss, learning_rate=learning_rate)
    evaluate = lstm.Evaluate(inference, label_placeholder)

    saver = tf.train.Saver()

    with tf.Session() as sess:
      if args.predict:
        print('Calculating prediction precision.')
        saver.restore(sess, checkpoint_path)
        total_test_precision = Evaluate(sess, batch_size, test_x, test_labels,
                                        x_placeholder, label_placeholder,
                                        evaluate)
        print('test_precision = %2.2f' % (total_test_precision * 100.0))
      else:
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

            feed_dict = {x_placeholder: batch_x,
                         label_placeholder: batch_label}

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            total_loss += loss_value

            duration = time.time() - start_time

            # print(lstm.final_cell.W_o.eval())
            # print(lstm.final_cell.embedding[1:100, :].eval())

          total_valid_precision = Evaluate(sess, batch_size, valid_x,
                                           valid_labels, x_placeholder,
                                           label_placeholder, evaluate)

          total_train_precision = Evaluate(sess, batch_size, x, labels,
                                           x_placeholder, label_placeholder,
                                           evaluate)

          print(
              'Epoch %d: loss = %.5f ; train_precision = %2.2f ; validation_precision = %2.2f (%.3f sec)'
              % (epoch, total_loss / i, total_train_precision * 100.0,
                 total_valid_precision * 100.0, duration))

          # Save the model.
          saver.save(sess, checkpoint_path)


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
