import tensorflow as tf
import tensorflow_compression as tfc
import functions
import rnn_cell_new

def one_step_rnn(tensor, state_c, state_h, Height, Width, num_filters, scale, kernal, act, spectralnorm=False):

    tensor = tf.expand_dims(tensor, axis=1)

    if not spectralnorm:
        cell = rnn_cell_new.ConvLSTMCell(shape=[Height // scale, Width // scale], activation=act,
                                     filters=num_filters, kernel=kernal)
    else:
        cell = rnn_cell_new.ConvLSTMCellSN(shape=[Height // scale, Width // scale], activation=act,
                                     filters=num_filters, kernel=kernal)
    state = tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
    tensor, state = tf.nn.dynamic_rnn(cell, tensor, initial_state=state, dtype=tensor.dtype)
    state_c, state_h = state

    tensor = tf.squeeze(tensor, axis=1)

    return tensor, state_c, state_h


def RAE(tensor, pre_func, post_func, num_filters, Height, Width, c_state, h_state, act):

    tensor = pre_func(tensor)

    with tf.variable_scope("recurrent"):
      tensor, c_state_out, h_state_out = one_step_rnn(tensor, c_state, h_state,
                                              Height, Width, num_filters,
                                              scale=4, kernal=[3, 3], act=act)
    tensor = post_func(tensor)

    return tensor, c_state_out, h_state_out


class AnaPre(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters=128, filter_size=(3, 3), *args, **kwargs):
    self.num_filters = num_filters
    self.filter_size = filter_size
    super(AnaPre, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
    ]
    super(AnaPre, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class AnaPost(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters=128, filter_size=(3, 3), *args, **kwargs):
    self.num_filters = num_filters
    self.filter_size = filter_size
    super(AnaPost, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2")),
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(AnaPost, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynPre(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters=128, filter_size=(3, 3), *args, **kwargs):
    self.num_filters = num_filters
    self.filter_size = filter_size
    super(SynPre, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
    ]
    super(SynPre, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynPost(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters=128, filter_size=(3, 3), output_ch = 2, *args, **kwargs):
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.output_ch = output_ch
    super(SynPost, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, self.filter_size, name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True)),
        tfc.SignalConv2D(
            self.output_ch, self.filter_size, name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynPost, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


def rec_density(tensor, num_filters, Height, Width, c_state, h_state, k=3, act=tf.tanh):


  with tf.variable_scope("CNN_input"):
      tensor = tf.expand_dims(tensor, axis=1)
      y1 = functions.recurrent_cnn(tensor, 1, layer=4, num_filters=num_filters, stride=1,
                                   out_filters=num_filters, kernel=[k, k], act=tf.nn.relu, act_last=None)
      y1 = tf.squeeze(y1, axis=1)

  with tf.variable_scope("RNN"):
      y2, c_state_out, h_state_out = one_step_rnn(y1, c_state, h_state,
                                                      Height, Width, num_filters,
                                                      scale=16, kernal=[k, k], act=act)

  with tf.variable_scope("CNN_output"):
      y2 = tf.expand_dims(y2, axis=1)
      y3 = functions.recurrent_cnn(y2, 1, layer=4, num_filters=num_filters, stride=1,
                                   out_filters=2 * num_filters, kernel=[k, k], act=tf.nn.relu, act_last=None)
      y3 = tf.squeeze(y3, axis=1)
  return y3, c_state_out, h_state_out


def bpp_est(x_target, sigma_mu, num_filters, tiny=1e-10):

    sigma, mu = tf.split(sigma_mu, [num_filters, num_filters], axis=-1)

    half = tf.constant(.5, dtype=tf.float32)

    upper = tf.math.add(x_target, half)
    lower = tf.math.add(x_target, -half)

    sig = tf.maximum(sigma, -7.0)
    upper_l = tf.sigmoid(tf.multiply((upper - mu), (tf.exp(-sig) + tiny)))
    lower_l = tf.sigmoid(tf.multiply((lower - mu), (tf.exp(-sig) + tiny)))
    p_element = upper_l - lower_l
    p_element = tf.clip_by_value(p_element, tiny, 1 - tiny)

    ent = -tf.log(p_element) / tf.log(2.0)
    bits = tf.math.reduce_sum(ent)

    return bits