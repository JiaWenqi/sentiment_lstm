import sys
import tensorflow as tf


class LSTMCell(object):
  """A single LSTM cell."""

  def __init__(self, idx, batch_size, emb_dim, x_placeholder, h_prev, C_prev,
               embedding, W_f, b_f, W_i, b_i, W_C, b_C, W_o, b_o):
    self.idx = idx
    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.x_placeholder = x_placeholder
    self.h_prev = h_prev
    self.C_prev = C_prev
    self.embedding = embedding
    self.W_f = W_f
    self.b_f = b_f
    self.W_i = W_i
    self.b_i = b_i
    self.W_C = W_C
    self.b_C = b_C
    self.W_o = W_o
    self.b_o = b_o

    x_embedding = tf.nn.embedding_lookup(self.embedding,
                                         self.x_placeholder[:, idx])

    # forget gate
    concat_input = tf.concat(1, [self.h_prev, x_embedding])
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
                    emb_dim=self.emb_dim,
                    x_placeholder=self.x_placeholder,
                    h_prev=self.h,
                    C_prev=self.C,
                    embedding=self.embedding,
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

  def __init__(self,
               length,
               batch_size,
               voc_size,
               emb_dim,
               num_class,
               pretrained_emb=None):
    self.length = length
    self.batch_size = batch_size
    self.voc_size = voc_size
    self.emb_dim = emb_dim
    self.num_class = num_class
    self.pretrained_emb = pretrained_emb

  def Inference(self, x_placeholder):
    h_init = tf.zeros([self.batch_size, self.num_class], name='h_t-1')
    C_init = tf.zeros([self.batch_size, self.num_class], name='C_prev')
    if self.pretrained_emb is not None:
      embedding = tf.constant(self.pretrained_emb, name='embedding')
    else:
      embedding = tf.Variable(
          tf.truncated_normal([self.voc_size, self.emb_dim]),
          name='embedding')
    W_f = tf.Variable(
        tf.truncated_normal([self.num_class + self.emb_dim, self.num_class]),
        name='W_f')
    b_f = tf.Variable(
        tf.truncated_normal([self.batch_size, self.num_class]),
        name='b_f')
    W_i = tf.Variable(
        tf.truncated_normal([self.num_class + self.emb_dim, self.num_class]),
        name='W_i')
    b_i = tf.Variable(
        tf.truncated_normal([self.batch_size, self.num_class]),
        name='b_i')
    W_C = tf.Variable(
        tf.truncated_normal([self.num_class + self.emb_dim, self.num_class]),
        name='W_C')
    b_C = tf.Variable(
        tf.truncated_normal([self.batch_size, self.num_class]),
        name='b_C')
    W_o = tf.Variable(
        tf.truncated_normal([self.num_class + self.emb_dim, self.num_class]),
        name='W_o')
    b_o = tf.Variable(
        tf.truncated_normal([self.batch_size, self.num_class]),
        name='b_o')

    self.cell_list = []
    cell = LSTMCell(idx=0,
                    batch_size=self.batch_size,
                    emb_dim=self.emb_dim,
                    x_placeholder=x_placeholder,
                    h_prev=h_init,
                    C_prev=C_init,
                    embedding=embedding,
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

  def Loss(self, inference, label_placeholder, l2_regularization_weight):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(inference,
                                                             label_placeholder)
    loss = tf.reduce_mean(entropy)
    for v in tf.trainable_variables():
      loss += l2_regularization_weight * tf.nn.l2_loss(v)
    return loss

  def Train(self, loss, learning_rate, clip_value_min, clip_value_max):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads_and_vars = [
        (tf.clip_by_value(g, clip_value_min, clip_value_max), v)
        for g, v in grads_and_vars
    ]
    train_op = optimizer.apply_gradients(clipped_grads_and_vars)

    return train_op

  def Evaluate(self, inference, label_placeholder):
    correct = tf.nn.in_top_k(inference, label_placeholder, 1)
    return tf.reduce_mean(tf.cast(correct, tf.float32))
