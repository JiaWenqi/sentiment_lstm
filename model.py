import math
import sys
import tensorflow as tf
import numpy as np


class LSTMCell(object):
  """A single LSTM cell."""

  def __init__(self, scope, keep_prob, is_training):
    self.scope = scope
    self.keep_prob = keep_prob
    self.is_training = is_training

  def __call__(self, x_placeholder, h_prev, C_prev):
    with tf.variable_scope(self.scope, reuse=True):
      embedding = tf.get_variable('embedding')
      W = tf.get_variable('weight')

    x_embedding = tf.nn.embedding_lookup(embedding, x_placeholder)

    if self.is_training:
      x_embedding = tf.nn.dropout(x_embedding, self.keep_prob)

    # forget gate
    concat_input = tf.concat(1, [h_prev, x_embedding])
    gates = tf.matmul(concat_input, W)
    m_f, m_i, m_C_update, m_o = tf.split(1, 4, gates)

    # forget gate
    f = tf.sigmoid(m_f)
    # input gate
    i = tf.sigmoid(m_i)
    # output gate
    o = tf.sigmoid(m_o)
    # Cell update
    C_update = tf.tanh(m_C_update)

    # cell after update
    # Add a dropout layer.
    C = tf.mul(f, C_prev) + tf.mul(i, C_update)

    # output
    h = tf.mul(o, tf.tanh(C))
    return h, C


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
    self.scope = 'lstm'

    def constant_embedding_initializer(shape=None, dtype=None):
      return self.pretrained_emb

    def ortho_weight(shape=None, dtype=None):
      dim = max(shape)
      W = np.random.randn(dim, dim)
      u, s, v = np.linalg.svd(W)
      return v[:shape[0], :shape[1]].astype(np.float32)

    with tf.variable_scope(self.scope):
      if self.pretrained_emb is not None:
        embedding = tf.get_variable('embedding',
                                    shape=[self.voc_size, self.emb_dim],
                                    initializer=constant_embedding_initializer,
                                    trainable=False)
      else:
        embedding = tf.get_variable(
            'embedding',
            shape=[self.voc_size, self.emb_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.01))

      W = tf.get_variable(
          'weight',
          shape=[self.state_size + self.emb_dim, 4 * self.state_size],
          initializer=ortho_weight)

      # logistic regression layer to convert from h to logits.
      W_h = tf.get_variable('weight_softmax',
                            shape=[self.state_size, self.num_class],
                            initializer=tf.truncated_normal_initializer(
                                stddev=math.sqrt(6.0 / self.state_size)))

      h_init = tf.get_variable('h_init',
                               shape=[self.batch_size, self.state_size],
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)
      C_init = tf.get_variable('C_init',
                               shape=[self.batch_size, self.state_size],
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)

  def Inference(self, x_placeholder, is_training=True):

    cell = LSTMCell(scope=self.scope,
                    keep_prob=self.keep_prob,
                    is_training=is_training)

    with tf.variable_scope(self.scope, reuse=True):
      W_h = tf.get_variable('weight_softmax',
                            shape=[self.state_size, self.num_class])

      h_init = tf.get_variable('h_init',
                               shape=[self.batch_size, self.state_size])
      C_init = tf.get_variable('C_init',
                               shape=[self.batch_size, self.state_size])

    h_prev = h_init
    C_prev = C_init

    cell_transition = tf.expand_dims(C_prev[14, :], 1)

    for i in range(self.length):
      h_prev, C_prev = cell(x_placeholder=x_placeholder[:, i],
                            h_prev=h_prev,
                            C_prev=C_prev)
      cell_transition = tf.concat(1, [cell_transition,
                                      tf.expand_dims(C_prev[14, :], 1)])

      # self.mean_h = tf.reduce_mean(
      #     tf.pack([cell.h for cell in self.cell_list]), 0)

    logits = tf.matmul(h_prev, W_h)

    return logits, tf.tanh(cell_transition)

  def Loss(self,
           inference,
           label_placeholder,
           l2_regularization_weight,
           name='training'):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(inference,
                                                             label_placeholder)
    loss = tf.reduce_sum(entropy)

    _ = tf.scalar_summary('%s: softmax cross entropy loss' % name, loss)
    # for v in tf.trainable_variables():
    #   loss += l2_regularization_weight * tf.nn.l2_loss(v)
    return loss

  def Train(self,
            loss,
            learning_rate,
            clip_value_min,
            clip_value_max,
            name='training'):
    tf.scalar_summary(':'.join([name, loss.op.name]), loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)

    clipped_grads_and_vars = [
        (tf.clip_by_value(g, clip_value_min, clip_value_max), v)
        for g, v in grads_and_vars
    ]

    for g, v in clipped_grads_and_vars:
      _ = tf.histogram_summary(':'.join([name, v.name]), v)
      _ = tf.histogram_summary('%s: gradient for %s' % (name, v.name), g)

    train_op = optimizer.apply_gradients(clipped_grads_and_vars)

    return train_op

  def Evaluate(self, inference, label_placeholder, name='training'):
    correct = tf.nn.in_top_k(inference, label_placeholder, 1)
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
    _ = tf.scalar_summary('%s: accuracy' % name,
                          accuracy / self.batch_size * 100.0)
    return accuracy, correct
