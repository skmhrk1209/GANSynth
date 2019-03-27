import tensorflow as tf
import numpy as np


def get_weight(shape, variance_scale=2, scale_weight=False):
    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
    if scale_weight:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.initializers.truncated_normal(0, 1)
        ) * stddev
    else:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.initializers.truncated_normal(0, stddev)
        )
    return weight


def get_bias(shape):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=tf.initializers.zeros()
    )
    return bias


def dense(inputs, units, use_bias=True, variance_scale=2, scale_weight=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d(inputs, filters, kernel_size, strides=[1, 1], use_bias=True,
           variance_scale=2, scale_weight=True):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    inputs = tf.nn.conv2d(
        input=inputs,
        filter=weight,
        strides=[1, 1] + strides,
        padding="SAME",
        data_format="NCHW"
    )
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def conv2d_transpose(inputs, filters, kernel_size, strides=[1, 1], use_bias=True,
                     variance_scale=2, scale_weight=True):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    weight = tf.transpose(weight, [0, 1, 3, 2])
    input_shape = np.array(inputs.shape)
    output_shape = [tf.shape(inputs)[0], filters, *input_shape[2:] * strides]
    inputs = tf.nn.conv2d_transpose(
        value=inputs,
        filter=weight,
        output_shape=output_shape,
        strides=[1, 1] + strides,
        padding="SAME",
        data_format="NCHW"
    )
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def upscale2d(inputs, factors=[2, 2]):
    factors = np.asanyarray(factors)
    if (factors == 1).all():
        return inputs
    shape = inputs.shape
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
    inputs = tf.tile(inputs, [1, 1, 1, factors[0], 1, factors[1]])
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2] * factors[0], shape[3] * factors[1]])
    return inputs


def downscale2d(inputs, factors=[2, 2]):
    # NOTE: requires tf_config["graph_options.place_pruned_graph"] = True
    factors = np.asanyarray(factors)
    if (factors == 1).all():
        return inputs
    inputs = tf.nn.avg_pool(
        value=inputs,
        ksize=[1, 1, *factors],
        strides=[1, 1, *factors],
        padding="SAME",
        data_format="NCHW"
    )
    return inputs


def embedding(inputs, units, variance_scale=2, scale_weight=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight
    )
    inputs = tf.nn.embedding_lookup(weight, tf.argmax(inputs, axis=1))
    return inputs


def pixel_norm(inputs, epsilon=1e-8):
    inputs *= tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
    return inputs


def batch_stddev(inputs, group_size=4, epsilon=1e-8):
    shape = inputs.shape
    inputs = tf.reshape(inputs, [group_size, -1, *shape[1:]])
    inputs -= tf.reduce_mean(inputs, axis=0, keepdims=True)
    inputs = tf.square(inputs)
    inputs = tf.reduce_mean(inputs, axis=0)
    inputs = tf.sqrt(inputs + epsilon)
    inputs = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
    inputs = tf.tile(inputs, [group_size, 1, *shape[2:]])
    return inputs
