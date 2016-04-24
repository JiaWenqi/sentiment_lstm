import unittest
import numpy as np
import tensorflow as tf

import model


class TestLSTM(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestLSTM, self).__init__(*args, **kwargs)
    self.learning_rate = 0.001
    self.keep_prob = 0.5
    self.num_class = 2
    self.state_size = 10
    self.pretrained_emb = None

  def tearDown(self):
    with tf.Graph().as_default():
      x_placeholder = tf.placeholder(tf.int32,
                                     shape=[self.batch_size, self.length],
                                     name='x')

      label_placeholder = tf.placeholder(tf.int64,
                                         shape=[self.batch_size,],
                                         name='label')

      lstm = model.LSTM(length=self.length,
                        batch_size=self.batch_size,
                        voc_size=self.voc_size,
                        emb_dim=self.emb_dim,
                        keep_prob=self.keep_prob,
                        num_class=self.num_class,
                        state_size=self.state_size,
                        pretrained_emb=self.pretrained_emb)
      inference, _ = lstm.Inference(x_placeholder)
      loss = lstm.Loss(inference, label_placeholder, 0.001)
      train_op = lstm.Train(loss, self.learning_rate, -0.1, 0.1)

      feed_dict = {x_placeholder: self.x, label_placeholder: self.label}
      with tf.Session() as sess:

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(10):
          _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

  def test_one_row_one_entry(self):
    self.length = 1
    self.batch_size = 1
    self.voc_size = 1000
    self.emb_dim = 5
    self.x = [[1]]
    self.label = [1]

  def test_one_row(self):
    self.length = 3
    self.batch_size = 1
    self.voc_size = 1000
    self.emb_dim = 100
    self.x = [[1, 2, 3]]
    self.label = [1]

  def test_multiple_row(self):
    self.length = 3
    self.batch_size = 3
    self.voc_size = 1000
    self.emb_dim = 100
    self.state_size = 20
    self.x = [[1, 2, 3], [1, 2, 4], [1, 3, 4]]
    self.label = [1, 0, 1]

  def test_multiple_row_pretrained_emb(self):
    self.length = 3
    self.batch_size = 3
    self.voc_size = 1000
    self.emb_dim = 100
    self.x = [[1, 2, 3], [1, 2, 4], [1, 3, 4]]
    self.label = [1, 0, 1]
    self.pretrained_emb = np.zeros(
        [self.voc_size, self.emb_dim],
        dtype=np.float32)


if __name__ == '__main__':
  unittest.main()
