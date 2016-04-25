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
  parser.add_argument('--analyze',
                      help='Analyze LSTM cell',
                      action='store_true')
  parser.add_argument('--error',
                      help='Print error prediction',
                      action='store_true')
  parser.add_argument('--print_weight',
                      help='Print softmax weight',
                      action='store_true')
  parser.add_argument('--emb', default='glove', type=str)
  args = parser.parse_args()

  batch_size = 100
  length = 100
  keep_prob = 0.5
  num_class = 2
  learning_rate = 0.05
  epochs = 100000
  voc_size = 100000
  emb_dim = 300
  state_size = 100
  clip_value_min = -1000.0
  clip_value_max = 1000.0
  l2_regularization_wegith = 0
  word_2_vec_emb_path = '../data/imdb.emb.pkl'
  glove_vec_emb_path = '../data/imdb.glove.emb.pkl'

  # Analysis.
  sequence_file = 'sequence.csv'
  activation_file = 'activation.csv'

  checkpoint_dir = './checkpoint'
  checkpoint_file = 'lstm'

  dictionary = load.load_dictionary_key_idx()
  train, valid, test = load.load_data(n_words=voc_size)
  x, labels = load.prepare_data(train[0], train[1], maxlen=length)
  print('There are %d training cases.' % len(labels))
  valid_x, valid_labels = load.prepare_data(valid[0], valid[1], maxlen=length)
  test_x, test_labels = load.prepare_data(test[0], test[1], maxlen=length)

  if args.emb == 'glove':
    pretrained_emb_path = glove_vec_emb_path
  elif args.emb == 'word2vec':
    pretrained_emb_path = word_2_vec_emb_path
  else:
    print('--emb should be one of glove, word2vec')
    exit(1)

  print('Loading embedding from %s' % pretrained_emb_path)
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
                      keep_prob=keep_prob,
                      num_class=num_class,
                      state_size=state_size,
                      pretrained_emb=pretrained_emb)
    inference, cell_transition = lstm.Inference(x_placeholder)
    inference_validate, cell_transition_validate = lstm.Inference(
        x_placeholder,
        is_training=False)
    loss = lstm.Loss(inference, label_placeholder, l2_regularization_wegith)
    loss_validate = lstm.Loss(inference_validate,
                              label_placeholder,
                              l2_regularization_wegith,
                              name='validate')
    train_op = lstm.Train(loss,
                          learning_rate=learning_rate,
                          clip_value_min=clip_value_min,
                          clip_value_max=clip_value_max)
    evaluate, correct = lstm.Evaluate(inference, label_placeholder)
    evaluate_validate, correct_validate = lstm.Evaluate(inference_validate,
                                                        label_placeholder,
                                                        name='validate')

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
      merged = tf.merge_all_summaries()
      writer = tf.train.SummaryWriter('/tmp/sentiment_log', sess.graph_def)

      if args.predict:
        print('Calculating prediction precision.')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        total_test_precision = Evaluate(sess, batch_size, test_x, test_labels,
                                        x_placeholder, label_placeholder,
                                        evaluate, inference)
        print('test_precision = %2.2f' % (total_test_precision * 100.0))
      elif args.error:
        print('Print sequences with wrong prediction.')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        batch_x, batch_label = load.NextMiniBatch(test_x, test_labels, 5,
                                                  batch_size)
        feed_dict = {x_placeholder: batch_x, label_placeholder: batch_label}
        correct_value = sess.run(correct, feed_dict=feed_dict)
        print('len of batch: %d' % batch_x.shape[0])
        print('\nFalse positive sequences.\n')
        for idx in range(batch_label.shape[0]):
          if int(batch_label[idx]) == 0 and int(correct_value[idx]) == 0:
            print(load.PrintSequence(dictionary, batch_x[idx]))
        print('\nFalse negative sequences.\n')
        for idx in range(batch_label.shape[0]):
          if int(batch_label[idx]) == 1 and int(correct_value[idx]) == 0:
            print(load.PrintSequence(dictionary, batch_x[idx]))
        print('\nTrue positive sequences.\n')
        for idx in range(batch_label.shape[0]):
          if int(batch_label[idx]) == 1 and int(correct_value[idx]) == 1:
            print(load.PrintSequence(dictionary, batch_x[idx]))
        print('\nTrue negative sequences.\n')
        for idx in range(batch_label.shape[0]):
          if int(batch_label[idx]) == 0 and int(correct_value[idx]) == 1:
            print(load.PrintSequence(dictionary, batch_x[idx]))
      elif args.print_weight:
        print('Print weight.')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        with tf.variable_scope('lstm', reuse=True):
          w_softmax = tf.get_variable('weight_softmax')
          # np.set_printoptions(threshold=np.nan)
          print(w_softmax.eval())
          np.savetxt('w_softmax.csv', w_softmax.eval(), delimiter=',')
      elif args.analyze:
        print('Analyzing LSTM cell.')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        batch_x, batch_label = load.NextMiniBatch(test_x, test_labels, 5,
                                                  batch_size)
        feed_dict = {x_placeholder: batch_x, label_placeholder: batch_label}
        inference_value, cell_transition_value = sess.run(
            [inference, cell_transition],
            feed_dict=feed_dict)

        idx = 83
        seq = batch_x[idx]
        label = batch_label[idx]

        # print(load.PrintSequence(dictionary, seq))
        print('label: %d' % label)
        print('infered label: %r' % inference_value[idx])
        load.WriteSequenceToCSV(sequence_file, state_size, dictionary, seq)
        print('sequences written to %s' % sequence_file)
        np.savetxt(activation_file,
                   np.asarray(cell_transition_value),
                   delimiter=',')
        print('activation written to %s' % activation_file)

        # np.set_printoptions(threshold=np.nan)
        # print(cell_transition_value[:,:])
      else:
        print('initializing all variables.')

        init = tf.initialize_all_variables()
        sess.run(init)

        k = 0
        for epoch in range(epochs):
          i = 0
          total_loss = 0
          start_time = time.time()
          while True:

            batch_x, batch_label = load.NextMiniBatch(x, labels, i, batch_size)
            i = i + 1
            if batch_x is None or batch_label is None:
              break

            feed_dict = {x_placeholder: batch_x,
                         label_placeholder: batch_label}

            _, loss_value, inference_value, summary_str = sess.run(
                [train_op, loss, inference, merged],
                feed_dict=feed_dict)

            writer.add_summary(summary_str, k)
            k += 1

            # print('loss_value: %.5f for inference: %r, label: %r' %
            #       (loss_value, inference_value, batch_label))
            total_loss += loss_value

          duration = time.time() - start_time

          # print(lstm.final_cell.W_o.eval())
          # print(lstm.final_cell.embedding[1:100, :].eval())

          total_valid_precision = Evaluate(
              sess, batch_size, valid_x, valid_labels, x_placeholder,
              label_placeholder, evaluate_validate, inference_validate)

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
