import unittest
import numpy as np
import tensorflow as tf

import model


class TestLSTM(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestLSTM, self).__init__(*args, **kwargs)
    self.learning_rate = 0.001

  def tearDown(self):
    with tf.Graph().as_default():
      x_placeholder = tf.placeholder(
          tf.float32,
          shape=[self.batch_size, self.length * self.input_dim],
          name='label')

      label_placeholder = tf.placeholder(
          tf.float32,
          shape=[self.batch_size, self.output_dim],
          name='label')

      lstm = model.LSTM(length=self.length,
                        batch_size=self.batch_size,
                        input_dim=self.input_dim,
                        output_dim=self.output_dim)
      inference = lstm.Inference(x_placeholder)
      loss = lstm.Loss(inference, label_placeholder)
      train_op = lstm.Train(loss, learning_rate=self.learning_rate)

      feed_dict = {x_placeholder: self.x, label_placeholder: self.label}
      with tf.Session() as sess:

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(10):
          _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

  def test_1D(self):
    self.length = 3
    self.batch_size = 1
    self.input_dim = 1
    self.output_dim = 1
    self.x = [[1, 2, 3]]
    self.label = [[1]]

  def test_3D(self):
    self.length = 3
    self.batch_size = 1
    self.input_dim = 3
    self.output_dim = 5
    self.x = [[1, 0, 0, 0, 1, 0, 0, 0, 1]]
    self.label = [[1, 0, 0, 0, 0]]

  def test_3D_longer_sequence(self):
    self.length = 4
    self.batch_size = 1
    self.input_dim = 3
    self.output_dim = 5
    self.x = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1]]
    self.label = [[1, 0, 0, 0, 0]]

  def test_numpy_array(self):
    self.length = 3
    self.batch_size = 1
    self.input_dim = 3
    self.output_dim = 5
    self.x = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
    self.label = np.array([[1, 0, 0, 0, 0]])

  def test_numpy_array_minibatch(self):
    self.length = 3
    self.batch_size = 2
    self.input_dim = 3
    self.output_dim = 5
    self.x = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 0, 1, 1, 0, 1, 0, 1]
                      ])
    self.label = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 1]])


if __name__ == '__main__':
  unittest.main()
