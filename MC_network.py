import tensorflow as tf
import numpy as np
# import tensorflow.contrib.slim as slim
# import tensorflow_compression as tfc
# import resnet

def resblock(input, IC, OC, name, reuse=tf.AUTO_REUSE):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = tf.layers.conv2d(inputs=l1, filters=np.minimum(IC, OC), kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l1', reuse=reuse)

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = tf.layers.conv2d(inputs=l2, filters=OC, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l2', reuse=reuse)

    if IC != OC:
        input = tf.layers.conv2d(inputs=input, filters=OC, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'map', reuse=reuse)

    return input + l2


def MC_new(input, reuse=tf.AUTO_REUSE):

    m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1', reuse=reuse)

    m2 = resblock(m1, 64, 64, name='mc2', reuse=reuse)

    m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

    m4 = resblock(m3, 64, 64, name='mc4', reuse=reuse)

    m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

    m6 = resblock(m5, 64, 64, name='mc6', reuse=reuse)

    m7 = resblock(m6, 64, 64, name='mc7', reuse=reuse)

    m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

    m8 = m4 + m8

    m9 = resblock(m8, 64, 64, name='mc9', reuse=reuse)

    m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

    m10 = m2 + m10

    m11 = resblock(m10, 64, 64, name='mc11', reuse=reuse)

    m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12', reuse=reuse)

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13', reuse=reuse)

    return m13


def MC_new2(flow, Y0, Y1_warp, out_filter=3, name=None, reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([flow, Y0, Y1_warp], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 64, 64, name='mc2')

        m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

        m4 = resblock(m3, 64, 64, name='mc4')

        m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

        m6 = resblock(m5, 64, 64, name='mc6')

        m7 = resblock(m6, 64, 64, name='mc7')

        m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

        m8 = m4 + m8

        m9 = resblock(m8, 64, 64, name='mc9')

        m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

        m10 = m2 + m10

        m11 = resblock(m10, 64, 64, name='mc11')

        m12 = tf.layers.conv2d(inputs=m11, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

        m12 = tf.nn.relu(m12, name='relu12')

        m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13',
                               reuse=reuse)

    return m13



def MC_res(flow, Y0, Y1_warp, out_filter=3, name=None, reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([flow, Y0, Y1_warp], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 64, 64, name='mc2')

        m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

        m4 = resblock(m3, 64, 64, name='mc4')

        m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

        m6 = resblock(m5, 64, 64, name='mc6')

        m7 = resblock(m6, 64, 64, name='mc7')

        m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

        m8 = m4 + m8

        m9 = resblock(m8, 64, 64, name='mc9')

        m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

        m10 = m2 + m10

        m11 = resblock(m10, 64, 64, name='mc11')

        m12 = tf.layers.conv2d(inputs=m11, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

    return m12 + Y1_warp


def MC_new_light(flow, Y0, Y1_warp, out_filter=3, name=None, reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([flow, Y0, Y1_warp], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 32, 32, name='mc2')

        # m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')
        #
        # m4 = resblock(m3, 64, 64, name='mc4')
        #
        # m9 = resblock(m4, 64, 64, name='mc9')
        #
        # m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])
        #
        # m10 = m2 + m10

        m11 = resblock(m2, 32, 32, name='mc11')

        m12 = tf.layers.conv2d(inputs=m11, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

    return m12 + Y1_warp


def MC_light(flow, Y0, Y1_warp, out_filter=3, name=None, reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([flow, Y0, Y1_warp], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 64, 64, name='mc2')

        m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

        m4 = resblock(m3, 64, 64, name='mc4')

        m9 = resblock(m4, 64, 64, name='mc9')

        m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

        m10 = m2 + m10

        m11 = resblock(m10, 64, 64, name='mc11') + m1

        m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

        m12 = tf.nn.relu(m12, name='relu12')

        m13 = tf.layers.conv2d(inputs=m12, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

    return m13


def MC_light_skip(flow, Y0, Y1_warp, out_filter=3, name=None, reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([flow, Y0, Y1_warp], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 64, 64, name='mc2')

        m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

        m4 = resblock(m3, 64, 64, name='mc4')

        m9 = resblock(m4, 64, 64, name='mc9')

        m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

        m10 = m2 + m10

        m11 = resblock(m10, 64, 64, name='mc11') + m1

        m12 = tf.layers.conv2d(inputs=m11, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

        # m12 = tf.nn.relu(m12, name='relu12')
        #
        # m13 = tf.layers.conv2d(inputs=m12, filters=out_filter, kernel_size=3, strides=1, padding='same',
        #                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

    return m12 + Y1_warp


def inloop(MC, res_hat, out_filter=3, name='inloop', reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([MC, res_hat], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 32, 32, name='mc2')

        m11 = resblock(m2, 32, 32, name='mc11') + m1

        m12 = tf.layers.conv2d(inputs=m11, filters=32, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

        m12 = tf.nn.relu(m12, name='relu12')

        m13 = tf.layers.conv2d(inputs=m12, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

        return m13


def inloop_skip(MC, res_hat, out_filter=3, name='inloop_skip', reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        input = tf.concat([MC, res_hat], axis=-1)

        m1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, 32, 32, name='mc2')

        m11 = resblock(m2, 32, 32, name='mc11')

        m12 = tf.layers.conv2d(inputs=m11, filters=32, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

        m12 = tf.nn.relu(m12, name='relu12')

        m13 = tf.layers.conv2d(inputs=m12, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

        return m13 + MC + res_hat
