import tensorflow as tf
import numpy as np


def get_weight(shape, variance_scale=2, scale_weight=True):
    std = np.sqrt(variance_scale / np.prod(shape[:-1]))
    if scale_weight:
        weight = tf.get_variable("weight", shape=shape, initializer=tf.initializers.random_normal()) * std
    else:
        weight = tf.get_variable("weight", shape=shape, initializer=tf.initializers.random_normal(0, std))
    return weight


def get_bias(shape):
    bias = tf.get_variable("bias", shape=shape, initializer=tf.initializers.zeros())
    return bias


def dense(inputs, units, use_bias=True, variance_scale=2, scale_weight=True):
    weight = get_weight([inputs.shape[1].value, units], variance_scale=variance_scale, scale_weight=scale_weight)
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d(inputs, filters, kernel_size, strides=[1, 1], use_bias=True, variance_scale=2, scale_weight=True):
    weight = get_weight([*kernel_size, inputs.shape[1].value, filters], variance_scale=variance_scale, scale_weight=scale_weight)
    inputs = tf.nn.conv2d(inputs, weight, strides=[1, 1] + strides, padding="SAME", data_format="NCHW")
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def conv2d_transpose(inputs, filters, kernel_size, strides=[1, 1], use_bias=True, variance_scale=2, scale_weight=True):
    weight = get_weight([*kernel_size, inputs.shape[1].value, filters], variance_scale=variance_scale, scale_weight=scale_weight)
    weight = tf.transpose(weight, [0, 1, 3, 2])
    output_shape = [tf.shape(inputs)[0], filters, *np.array(inputs.shape.as_list()[2:]) * strides]
    inputs = tf.nn.conv2d_transpose(inputs, weight, output_shape, strides=[1, 1] + strides, padding="SAME", data_format="NCHW")
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def pixel_normalization(inputs, epsilon=1e-8):
    inputs *= tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
    return inputs


def upscale2d(inputs, factors=[2, 2]):
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
    inputs = tf.tile(inputs, [1, 1, 1, factors[0], 1, factors[1]])
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2] * factors[0], shape[3] * factors[1]])
    return inputs


def downscale2d(inputs, factors=[2, 2]):
    # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True
    inputs = tf.nn.avg_pool(inputs, ksize=[1, 1, *factors], strides=[1, 1, *factors], padding="SAME", data_format='NCHW')
    return inputs


def batch_stddev(inputs, group_size=4):
    shape = inputs.shape.as_list()
    stddev = tf.reshape(inputs, [group_size, -1, *shape[1:]])
    stddev -= tf.reduce_mean(stddev, axis=0, keepdims=True)
    stddev = tf.reduce_mean(tf.square(stddev), axis=0)
    stddev = tf.sqrt(stddev + 1e-8)
    stddev = tf.reduce_mean(stddev, axis=[1, 2, 3], keepdims=True)
    stddev = tf.tile(stddev, [group_size, 1, *shape[2:]])
    inputs = tf.concat([inputs, stddev], axis=1)
    return inputs
