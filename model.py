import tensorflow as tf


class LSTMCell(object):
  """A single LSTM cell."""

  def __init__(self, batch_size, input_dim, h_prev, C_prev, W_f, b_f, W_i, b_i,
               W_C, b_C, W_o, b_o):
    self.batch_size = batch_size
    self.input_dim = input_dim
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

    self.x_placeholder = tf.placeholder(tf.float32,
                                        shape=[self.batch_size, self.input_dim])

    # forget gate
    concat_input = tf.concat(1, [self.h_prev, self.x_placeholder])
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

  def Next(self):
    return LSTMCell(batch_size=self.batch_size,
                    input_dim=self.input_dim,
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

    self.label_placeholder = tf.placeholder(
        tf.float32,
        shape=[self.batch_size, self.output_dim],
        name='label')
    h_init = tf.Variable(
        tf.truncated_normal([self.batch_size, self.output_dim]),
        name='h_t-1')
    C_init = tf.Variable(
        tf.truncated_normal([self.batch_size, self.output_dim]),
        name='C_prev')
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
    cell = LSTMCell(batch_size=self.batch_size,
                    input_dim=self.input_dim,
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

  @property
  def inference(self):
    return self.final_cell.h

  @property
  def loss(self):
    # x-y
    diff = self.inference - self.label_placeholder
    # (x-y)^2
    diff_pow = tf.pow(diff, 2)
    # Sigma((x-y)^2) for each run
    diff_pow_batch = tf.reduce_sum(diff_pow, 1)
    # sqrt(sigma(..)) for each run
    euclidean_distance = tf.sqrt(diff_pow_batch)
    # mean for the whole batch
    return tf.reduce_mean(euclidean_distance)

  def Train(self, loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

  @property
  def inputs(self):
    return [cell.x_placeholder for cell in self.cell_list]
