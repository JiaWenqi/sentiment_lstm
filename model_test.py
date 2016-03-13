import unittest
import numpy
import tensorflow as tf

import model


class TestLSTM(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestLSTM, self).__init__(*args, **kwargs)
    self.learning_rate = 0.001

  def tearDown(self):
    with tf.Graph().as_default():
      lstm = model.LSTM(length=len(self.x),
                        batch_size=len(self.x[0]),
                        input_dim=len(self.x[0][0]),
                        output_dim=len(self.label[0]))
      loss = lstm.loss
      train_op = lstm.Train(loss, learning_rate=self.learning_rate)

      feed_dict_x = list(zip(lstm.inputs, self.x))
      feed_dict_label = {lstm.label_placeholder: self.label}

      feed_dict = {}
      feed_dict.update(feed_dict_x)
      feed_dict.update(feed_dict_label)

      with tf.Session() as sess:

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(10):
          _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

  def test_1D(self):
    self.x = [[[1]], [[2]], [[3]]]
    self.label = [[1]]

  def test_3D(self):
    self.x = [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]]
    self.label = [[1, 0, 0, 0, 0]]

  def test_3D_longer_sequence(self):
    self.x = [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[0, 1, 1]]]
    self.label = [[1, 0, 0, 0, 0]]


if __name__ == '__main__':
  unittest.main()
