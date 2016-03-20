import sys
import tensorflow as tf


class LSTMCell(object):
  """A single LSTM cell."""

  def __init__(self, idx, batch_size, input_dim, x_placeholder, h_prev, C_prev,
               W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o):
    self.idx = idx
    self.batch_size = batch_size
    self.input_dim = input_dim
    self.x_placeholder = x_placeholder
    self.h_prev = h_prev
    self.C_prev = C_prev
    self.W_f = W_f
    self.b_f = b_f
    self.W_i = W_i
    self.b_i = b_i
    self.W_C = W_C
    self.b_C = b_C
    self.W_o = W_o
    self.b_o = b_o

    # forget gate
    concat_input = tf.concat(1, [
        self.h_prev,
        self.x_placeholder[:, idx * self.input_dim:(idx + 1) * self.input_dim]
    ])
    f = tf.sigmoid(tf.matmul(concat_input, self.W_f) + self.b_f)

    # input gate
    i = tf.sigmoid(tf.matmul(concat_input, self.W_i) + self.b_i)

    # cell update
    C_update = tf.tanh(tf.matmul(concat_input, self.W_C) + self.b_C)

    # cell after update
    self.C = tf.mul(f, self.C_prev) + tf.mul(i, C_update)

    # output gate
    o = tf.sigmoid(tf.matmul(concat_input, self.W_o) + self.b_o)

    # output
    self.h = tf.mul(o, tf.tanh(self.C))

    self.concat_input = concat_input

  def Next(self):
    return LSTMCell(idx=self.idx + 1,
                    batch_size=self.batch_size,
                    input_dim=self.input_dim,
                    x_placeholder=self.x_placeholder,
                    h_prev=self.h,
                    C_prev=self.C,
                    W_f=self.W_f,
                    b_f=self.b_f,
                    W_i=self.W_i,
                    b_i=self.b_i,
                    W_C=self.W_C,
                    b_C=self.b_C,
                    W_o=self.W_o,
                    b_o=self.b_o)


class LSTM(object):
  """A composite LSTM made of LSTM cells."""

  def __init__(self, length, batch_size, input_dim, output_dim):
    self.length = length
    self.batch_size = batch_size
    self.input_dim = input_dim
    self.output_dim = output_dim

  def Inference(self, x_placeholder):
    h_init = tf.zeros([self.batch_size, self.output_dim], name='h_t-1')
    C_init = tf.zeros([self.batch_size, self.output_dim], name='C_prev')
    W_f = tf.Variable(
        tf.truncated_normal([self.output_dim + self.input_dim, self.output_dim
                            ]),
        name='W_f')
    b_f = tf.Variable(
        tf.truncated_normal([self.batch_size, self.output_dim]),
        name='b_f')
    W_i = tf.Variable(
        tf.truncated_normal([self.output_dim + self.input_dim, self.output_dim
                            ]),
        name='W_i')
    b_i = tf.Variable(
        tf.truncated_normal([self.batch_size, self.output_dim]),
        name='b_i')
    W_C = tf.Variable(
        tf.truncated_normal([self.output_dim + self.input_dim, self.output_dim
                            ]),
        name='W_C')
    b_C = tf.Variable(
        tf.truncated_normal([self.batch_size, self.output_dim]),
        name='b_C')
    W_o = tf.Variable(
        tf.truncated_normal([self.output_dim + self.input_dim, self.output_dim
                            ]),
        name='W_o')
    b_o = tf.Variable(
        tf.truncated_normal([self.batch_size, self.output_dim]),
        name='b_o')

    self.cell_list = []
    cell = LSTMCell(idx=0,
                    batch_size=self.batch_size,
                    input_dim=self.input_dim,
                    x_placeholder=x_placeholder,
                    h_prev=h_init,
                    C_prev=C_init,
                    W_f=W_f,
                    b_f=b_f,
                    W_i=W_i,
                    b_i=b_i,
                    W_C=W_C,
                    b_C=b_C,
                    W_o=W_o,
                    b_o=b_o)

    self.cell_list.append(cell)

    for i in range(1, self.length):
      cell = cell.Next()
      self.cell_list.append(cell)

    self.final_cell = self.cell_list[-1]

    return self.final_cell.h

  def Loss(self, inference, label_placeholder):
    # x-y
    diff = inference - label_placeholder
    # (x-y)^2
    diff_pow = tf.pow(diff, 2)
    # Sigma((x-y)^2) for each run
    diff_pow_batch = tf.clip_by_value(
        tf.reduce_sum(diff_pow, 1), 0.0000001, sys.maxint)
    # sqrt(sigma(..)) for each run
    euclidean_distance = tf.sqrt(diff_pow_batch)
    # mean for the whole batch
    distance = tf.reduce_mean(euclidean_distance)

    regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    return (distance + regularizer(self.final_cell.W_f) +
            regularizer(self.final_cell.W_i) + regularizer(self.final_cell.W_C)
            + regularizer(self.final_cell.W_o))

  def Train(self, loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

  def Evaluate(self, inference, label_placeholder):
    # x-y
    diff = inference - label_placeholder
    # (x-y)^2
    diff_pow = tf.pow(diff, 2)
    # Sigma((x-y)^2) for each run
    diff_pow_batch = tf.reduce_sum(diff_pow, 1)
    # sqrt(sigma(..)) for each run
    euclidean_distance = tf.sqrt(diff_pow_batch)
    bound = tf.fill(tf.shape(euclidean_distance), 0.5)
    cond = tf.less(euclidean_distance, bound)
    return tf.reduce_mean(tf.cast(cond, tf.float32))
