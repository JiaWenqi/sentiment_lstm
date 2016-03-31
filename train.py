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
  learning_rate = 0.01
  epochs = 100000
  voc_size = 100000
  emb_dim = 300
  state_size = 20
  clip_value_min = -5.0
  clip_value_max = 5.0
  l2_regularization_wegith = 0
  pretrained_emb_path = '../data/imdb.emb.pkl'
  checkpoint_dir = './checkpoint'
  checkpoint_file = 'lstm'

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
                      state_size=state_size,
                      pretrained_emb=pretrained_emb)
    inference = lstm.Inference(x_placeholder)
    loss = lstm.Loss(inference, label_placeholder, l2_regularization_wegith)
    train_op = lstm.Train(loss,
                          learning_rate=learning_rate,
                          clip_value_min=clip_value_min,
                          clip_value_max=clip_value_max)
    evaluate = lstm.Evaluate(inference, label_placeholder)

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
      if args.predict:
        print('Calculating prediction precision.')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
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

            _, loss_value, inference_value = sess.run(
                [train_op, loss, inference],
                feed_dict=feed_dict)

            # print('loss_value: %.5f for inference: %r, label: %r' %
            #       (loss_value, inference_value, batch_label))
            total_loss += loss_value

            duration = time.time() - start_time

            # print(lstm.final_cell.W_o.eval())
            # print(lstm.final_cell.embedding[1:100, :].eval())

          total_valid_precision = Evaluate(
              sess, batch_size, valid_x, valid_labels, x_placeholder,
              label_placeholder, evaluate, inference)

          total_train_precision = Evaluate(sess, batch_size, x, labels,
                                           x_placeholder, label_placeholder,
                                           evaluate, inference)

          print(
              'Epoch %d: loss = %.5f ; train_precision = %2.2f ; validation_precision = %2.2f (%.3f sec)'
              % (epoch, total_loss / (len(x) / batch_size * batch_size),
                 total_train_precision * 100.0, total_valid_precision * 100.0,
                 duration))

          # Save the model.
          saver.save(sess,
                     os.path.join(checkpoint_dir, checkpoint_file),
                     global_step=epoch)


def Evaluate(sess, batch_size, batch_x, batch_label, x_placeholder,
             label_placeholder, evaluate, inference):
  inference_value_list = []
  total_eval = 0.0
  j = 0
  while True:
    batch_test_x, batch_test_label = load.NextMiniBatch(batch_x, batch_label, j,
                                                        batch_size)
    j += 1
    if batch_test_x is None or batch_test_label is None:
      break

    feed_dict = {x_placeholder: batch_test_x,
                 label_placeholder: batch_test_label}

    inference_value, evaluate_value = sess.run(
        [inference, evaluate],
        feed_dict=feed_dict)

    inference_value_list.append(inference_value)
    total_eval += evaluate_value

  # print('inference:')
  # print(inference_value_list)
  # print('actual')
  # print(batch_label)

  return total_eval / (len(batch_x) / batch_size * batch_size)


if __name__ == '__main__':
  main()
