import math
import sys
import tensorflow as tf


class LSTMCell(object):
  """A single LSTM cell."""

  def __init__(self, idx, batch_size, emb_dim, keep_prob, x_placeholder, h_prev,
               C_prev, embedding, W_f, W_i, W_C, W_o):
    self.idx = idx
    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.keep_prob = keep_prob
    self.x_placeholder = x_placeholder
    self.h_prev = h_prev
    self.C_prev = C_prev
    self.embedding = embedding
    self.W_f = W_f
    self.W_i = W_i
    self.W_C = W_C
    self.W_o = W_o

    x_embedding = tf.nn.embedding_lookup(self.embedding,
                                         self.x_placeholder[:, idx])

    # forget gate
    concat_input = tf.concat(1, [self.h_prev, x_embedding])
    f = tf.sigmoid(tf.matmul(concat_input, self.W_f))

    # input gate
    i = tf.sigmoid(tf.matmul(concat_input, self.W_i))

    # cell update
    C_update = tf.tanh(tf.matmul(concat_input, self.W_C))

    # cell after update
    # Add a dropout layer.
    self.C = tf.nn.dropout(
        tf.mul(f, self.C_prev) + tf.mul(i, C_update), self.keep_prob)

    # output gate
    o = tf.sigmoid(tf.matmul(concat_input, self.W_o))

    # output
    self.h = tf.mul(o, tf.tanh(self.C))

    self.concat_input = concat_input

  def Next(self):
    return LSTMCell(idx=self.idx + 1,
                    batch_size=self.batch_size,
                    emb_dim=self.emb_dim,
                    keep_prob=self.keep_prob,
                    x_placeholder=self.x_placeholder,
                    h_prev=self.h,
                    C_prev=self.C,
                    embedding=self.embedding,
                    W_f=self.W_f,
                    W_i=self.W_i,
                    W_C=self.W_C,
                    W_o=self.W_o)


class LSTM(object):
  """A composite LSTM made of LSTM cells."""

  def __init__(self,
               length,
               batch_size,
               voc_size,
               emb_dim,
               keep_prob,
               num_class,
               state_size,
               pretrained_emb=None):
    self.length = length
    self.batch_size = batch_size
    self.voc_size = voc_size
    self.emb_dim = emb_dim
    self.keep_prob = keep_prob
    self.num_class = num_class
    self.state_size = state_size
    self.pretrained_emb = pretrained_emb

  def Inference(self, x_placeholder):
    h_init = tf.zeros([self.batch_size, self.state_size], name='h_t-1')
    C_init = tf.zeros([self.batch_size, self.state_size], name='C_prev')
    if self.pretrained_emb is not None:
      embedding = tf.constant(self.pretrained_emb, name='embedding')
    else:
      embedding = tf.Variable(
          tf.truncated_normal([self.voc_size, self.emb_dim]),
          name='embedding')
    W_f = tf.Variable(
        tf.truncated_normal(
            [self.state_size + self.emb_dim, self.state_size],
            stddev=1.0 / math.sqrt(self.state_size + self.emb_dim)),
        name='W_f')
    W_i = tf.Variable(
        tf.truncated_normal(
            [self.state_size + self.emb_dim, self.state_size],
            stddev=1.0 / math.sqrt(self.state_size + self.emb_dim)),
        name='W_i')
    W_C = tf.Variable(
        tf.truncated_normal(
            [self.state_size + self.emb_dim, self.state_size],
            stddev=1.0 / math.sqrt(self.state_size + self.emb_dim)),
        name='W_C')
    W_o = tf.Variable(
        tf.truncated_normal(
            [self.state_size + self.emb_dim, self.state_size],
            stddev=1.0 / math.sqrt(self.state_size + self.emb_dim)),
        name='W_o')

    # logistic regression layer to convert from h to logits.
    self.W_h = tf.Variable(
        tf.truncated_normal(
            [self.state_size, self.num_class],
            stddev=1.0 / math.sqrt(self.state_size)),
        name='W_h')

    _ = tf.histogram_summary('W_forget', W_f)
    _ = tf.histogram_summary('W_input', W_i)
    _ = tf.histogram_summary('W_output', W_o)
    _ = tf.histogram_summary('W_classify', self.W_h)

    self.cell_list = []

    cell = LSTMCell(idx=0,
                    batch_size=self.batch_size,
                    emb_dim=self.emb_dim,
                    keep_prob=self.keep_prob,
                    x_placeholder=x_placeholder,
                    h_prev=h_init,
                    C_prev=C_init,
                    embedding=embedding,
                    W_f=W_f,
                    W_i=W_i,
                    W_C=W_C,
                    W_o=W_o)

    self.cell_list.append(cell)

    for i in range(1, self.length):
      cell = cell.Next()
      self.cell_list.append(cell)

    self.final_cell = self.cell_list[-1]
    # self.mean_h = tf.reduce_mean(
    #     tf.pack([cell.h for cell in self.cell_list]), 0)

    logits = tf.matmul(self.final_cell.h, self.W_h)

    return logits

  def Loss(self, inference, label_placeholder, l2_regularization_weight):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(inference,
                                                             label_placeholder)
    loss = tf.reduce_sum(entropy)

    _ = tf.scalar_summary('softmax cross entropy loss', loss)
    # for v in tf.trainable_variables():
    #   loss += l2_regularization_weight * tf.nn.l2_loss(v)
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
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
    _ = tf.scalar_summary('accuracy', accuracy)
    return accuracy
