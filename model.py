import tensorflow as tf


class LSTMCell(object):
  """http://colah.github.io/posts/2015-08-Understanding-LSTMs/
  """

  def __init__(self, input_width, prev_h, W_forget, b_forget, W_input, b_input,
               W_C, b_C, C_prev, W_output, b_output):
    self.input_width = input_width
    self.W_forget = W_forget
    self.b_forget = b_forget
    self.W_input = W_input
    self.b_input = b_input
    self.W_C = W_C
    self.b_C = b_C
    self.W_output = W_output
    self.b_output = b_output

    self.input = tf.placeholder(dtype=tf.int32,
                                shape=(input_width),
                                name='input')
    self.label = tf.placeholder(dtype=tf.int32,
                                shape=(input_width),
                                name='label')
    prev_h_and_w = tf.concat(1, [prev_h, self.input])

    forget_gate = tf.sigmoid(tf.matmul(W_forget, prev_h_and_w) + b_forget)

    input_gate = tf.sigmoid(tf.matmul(W_input, prev_h_and_w) + b_input)

    C_tilt_t = tf.tanh(tf.matmul(W_C, prev_h_and_w) + b_C)

    self.C_t = forget_gate * C_prev + input_gate * C_tilt_t

    output_gate = tf.sigmoid(tf.matmul(W_output, prev_h_and_w) + b_output)
    self.h = output_gate * tf.tanh(self.C_t)

  def Stack(self):
    return LSTMCell(self.input_width, self.h, self.W_forget, self.b_forget,
                    self.W_input, self.b_input, self.W_C, self.b_C, self.C_t,
                    self.W_output, self.b_output)

  def Inference(self):
    return self.h

  def Loss(self):
    return


class LSTM(object):

  def __init__(self, length, input_width, h_width):
    self.length = length

    W_shape = [input_width + h_width, 1]
    b_shape = [1, 1]
    prev_h = tf.Variable(tf.random_normal([1, h_width]))
    W_forget = tf.Variable(tf.random_normal(W_shape))
    b_forget = tf.Variable(tf.random_normal(b_shape))
    W_input = tf.Variable(tf.random_normal(W_shape))
    b_forget = tf.Variable(tf.random_normal(b_shape))
    W_C = tf.Variable(tf.random_normal(W_shape))
    b_C = tf.Variable(tf.random_normal(b_shape))
    C_prev = tf.Variable(tf.random_normal([1, h_width]))
    W_output = tf.Variable(tf.random_normal(W_shape))
    b_output = tf.Variable(tf.random_normal(b_shape))

    cell_list = []
    cell = LSTMCell(input_width, prev_h, W_forget, b_forget, W_input, b_input,
                    W_C, b_C, C_prev, W_output, b_output)

    cell_list.append(cell)

    for i in range(1, length):
      cell = cell.Stack()
      cell_list.append(cell)

  def Inference(self):
    return [cell.Inference() for cell in cell_list]

  def Loss(self):
    return
