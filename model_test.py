import unittest
import numpy
import tensorflow as tf

import model


class TestLSTM(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestLSTM, self).__init__(*args, **kwargs)

  def test_Trivial(self):
    with tf.Graph().as_default():
      lstm = model.LSTM(length=3, batch_size=1, input_dim=1, output_dim=1)
      loss = lstm.loss
      train_op = lstm.Train(loss, learning_rate=0.001)

      x = [[[1]], [[2]], [[3]]]
      label = [[1]]

      feed_dict_x = list(zip(lstm.inputs, x))
      feed_dict_label = {lstm.label_placeholder: label}

      feed_dict = {}
      feed_dict.update(feed_dict_x)
      feed_dict.update(feed_dict_label)

      with tf.Session() as sess:

        init = tf.initialize_all_variables()
        sess.run(init)

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
