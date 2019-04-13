import tensorflow as tf
import numpy as np


def weight_standardization(weight, epsilon=1.0e-8):
    shape = weight.shape.as_list()
    weight = tf.reshape(weight, [-1, shape[-1]])
    mean, variance = tf.nn.moments(weight, axes=[0], keep_dims=True)
    std = tf.sqrt(variance + epsilon)
    weight = (weight - mean) / std
    weight = tf.reshape(weight, shape)
    return weight


def group_normalization(inputs, groups, epsilon=1.0e-8):
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, groups, shape[1] // groups, *shape[2:]])
    mean, variance = tf.nn.moments(inputs, axes=[2, 3, 4], keep_dims=True)
    std = tf.sqrt(variance + epsilon)
    inputs = (inputs - mean) / std
    inputs = tf.reshape(inputs, [-1, *shape[1:]])
    gamma = tf.get_variable(
        name="gamma",
        shape=[1, shape[1], 1, 1],
        initializer=tf.initializers.ones()
    )
    beta = tf.get_variable(
        name="beta",
        shape=[1, shape[1], 1, 1],
        initializer=tf.initializers.zeros()
    )
    inputs = inputs * gamma + beta
    return inputs


def get_weight(shape, variance_scale=2.0, scale_weight=False, apply_weight_standardization=False):
    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
    if scale_weight:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.initializers.truncated_normal(0.0, 1.0)
        ) * stddev
    else:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.initializers.truncated_normal(0.0, stddev)
        )
    if apply_weight_standardization:
        weight = weight_standardization(weight)
    return weight


def get_bias(shape):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=tf.initializers.zeros()
    )
    return bias


def dense(inputs, units, use_bias=True, variance_scale=2.0, scale_weight=False, apply_weight_standardization=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def embedding(inputs, units, variance_scale=2.0, scale_weight=False, apply_weight_standardization=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization
    )
    inputs = tf.nn.embedding_lookup(weight, tf.argmax(inputs, axis=1))
    return inputs


def conv2d(inputs, filters, kernel_size, strides=[1, 1], use_bias=True,
           variance_scale=2.0, scale_weight=False, apply_weight_standardization=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization
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
                     variance_scale=2.0, scale_weight=False, apply_weight_standardization=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization
    )
    weight = tf.transpose(weight, [0, 1, 3, 2])
    input_shape = np.array(inputs.shape.as_list())
    output_shape = [input_shape[0], filters, *input_shape[2:] * strides]
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
    shape = inputs.shape.as_list()
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


def pixel_norm(inputs, epsilon=1.0e-8):
    inputs /= tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
    return inputs


def batch_stddev(inputs, groups=4, epsilon=1.0e-8):
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [groups, -1, *shape[1:]])
    inputs -= tf.reduce_mean(inputs, axis=0, keepdims=True)
    inputs = tf.square(inputs)
    inputs = tf.reduce_mean(inputs, axis=0)
    inputs = tf.sqrt(inputs + epsilon)
    inputs = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
    inputs = tf.tile(inputs, [groups, 1, *shape[2:]])
    return inputs


def max_pooling2d(inputs, kernel_size, strides):
    inputs = tf.nn.max_pool(
        value=inputs,
        ksize=[1, 1, *kernel_size],
        strides=[1, 1, *strides],
        padding="SAME",
        data_format="NCHW"
    )
    return inputs


def average_pooling2d(inputs, kernel_size, strides):
    inputs = tf.nn.avg_pool(
        value=inputs,
        ksize=[1, 1, *kernel_size],
        strides=[1, 1, *strides],
        padding="SAME",
        data_format="NCHW"
    )
    return inputs
