import tensorflow as tf
import numpy as np
from scipy import misc

def cnn_layers(tensor, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for l in range(layer-1):

       tensor = tf.layers.conv2d(inputs=tensor, filters=num_filters, kernel_size=kernel, padding='same',
                       reuse=reuse, activation=act, strides=stride,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='cnn_' + str(l + 1))

    tensor = tf.layers.conv2d(inputs=tensor, filters=out_filters, kernel_size=kernel, padding='same',
                    reuse=reuse, activation=act_last, strides=stride,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='cnn_' + str(layer))

    return tensor


def dnn_layers(tensor, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for l in range(layer-1):

       tensor = tf.layers.conv2d_transpose(inputs=tensor, filters=num_filters, kernel_size=kernel, padding='same',
                       reuse=reuse, activation=act, strides=stride,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='dnn_' + str(l + 1))

    tensor = tf.layers.conv2d_transpose(inputs=tensor, filters=out_filters, kernel_size=kernel, padding='same',
                    reuse=reuse, activation=act_last, strides=stride,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=uni), name='dnn_' + str(layer))

    return tensor


def recurrent_cnn(tensor, step, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for i in range(step):

        tensor_i = tensor[:, i, :, :, :]
        tensor_i = cnn_layers(tensor_i, layer, num_filters, out_filters, kernel, stride, uni, act, act_last, reuse)

        if i == 0:
            tensor_out = tf.expand_dims(tensor_i, 1)
        else:
            tensor_out = tf.concat([tensor_out, tf.expand_dims(tensor_i, 1)], axis=1)

    return tensor_out


def recurrent_dnn(tensor, step, layer, num_filters, out_filters, kernel, stride=2, uni=True, act=tf.nn.relu, act_last=None, reuse=tf.AUTO_REUSE):

    for i in range(step):

        tensor_i = tensor[:, i, :, :, :]
        tensor_i = dnn_layers(tensor_i, layer, num_filters, out_filters, kernel, stride, uni, act, act_last, reuse)

        if i == 0:
            tensor_out = tf.expand_dims(tensor_i, 1)
        else:
            tensor_out = tf.concat([tensor_out, tf.expand_dims(tensor_i, 1)], axis=1)

    return tensor_out


def mse_tf(A, B):

    return tf.reduce_mean(tf.squared_difference(A, B))


def psnr_tf(A, B):

    mse = tf.reduce_mean(tf.squared_difference(A, B))
    psnr = 10.0*tf.log(1.0/mse)/tf.log(10.0)

    return psnr


def mse_np(A, B):

    return np.mean(np.power(np.subtract(A, B), 2.0))


def psnr_np(A, B):

    mse = np.mean(np.power(np.subtract(A, B), 2.0))
    psnr = 10.0*np.log(1.0/mse)/np.log(10.0)

    return psnr


def load_data(data, step, batch_size, Height, Width, folder):

    for b in range(batch_size):

        path = folder[np.random.randint(1, len(folder))] + '/'
        bb = np.random.randint(0, 448 - Width)

        for s in range(step):

            img = misc.imread(path + 'im' + str(s + 1) + '.png')
            data[b, s, 0:Height, 0:Width, :] = img[0:Height, bb: bb + Width, :] / 255.0

    return data

def load_data_P(data, step, batch_size, Height, Width, folder, q, bpg=True):

    for b in range(batch_size):

        path = folder[np.random.randint(1, len(folder))] + '/'
        bb = np.random.randint(0, 448 - Width)

        # img = misc.imread(path + 'im' + str(1) + '.png')
        # data_I[b, 0:Height, 0:Width, :] = img[0:Height, bb: bb + Width, :] / 255.0

        for s in range(step):

            if s == 0:
                if bpg:
                    img = misc.imread(path + 'im' + str(s + 1) + '_bpg' + str(q) + '_444.png')
                else:
                    img = misc.imread(path + 'im' + str(s + 1) + '_psnr_' + str(q) + '.png')
            else:
                img = misc.imread(path + 'im' + str(s + 1) + '.png')

            data[b, s, 0:Height, 0:Width, :] = img[0:Height, bb: bb + Width, :] / 255.0

    return data



