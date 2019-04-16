import tensorflow as tf
import numpy as np


def assign_moving_average(variable, value, momentum):
    assigned = tf.assign_sub(variable, (variable - value) * (1.0 - momentum))
    return assigned


def spectral_normalization(weight, iterations=1, epsilon=1.0e-12):
    shape = weight.shape.as_list()
    w = tf.reshape(weight, [-1, shape[-1]])
    # Persist the first right singular vector.
    u_var = tf.get_variable(
        name="u_var",
        shape=[1, shape[-1]],
        initializer=tf.initializers.random_normal(),
        trainable=False
    )
    u = u_var
    # Use power iteration method to approximate the spectral norm.
    for _ in range(iterations):
        v = tf.matmul(u, w, transpose_b=True)
        v = tf.nn.l2_normalize(v, epsilon=epsilon)
        u = tf.matmul(v, w)
        u = tf.nn.l2_normalize(u, epsilon=epsilon)
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)
    spectral_norm = tf.matmul(tf.matmul(v, w), u, transpose_b=True)
    weight = weight / spectral_norm
    # Update the approximation.
    with tf.control_dependencies([tf.assign(u_var, u)]):
        weight = tf.indentity(weight)
    return weight


def weight_standardization(weight, epsilon=1.0e-12):
    shape = weight.shape.as_list()
    mean, variance = tf.nn.moments(
        x=weight,
        axes=list(range(0, len(shape) - 1)),
        keep_dims=True
    )
    stddev = tf.sqrt(variance + epsilon)
    weight = (weight - mean) / stddev
    return weight


def batch_normalization(inputs, training, momentum=0.99, epsilon=1.0e-12):
    training = tf.convert_to_tensor(training)
    shape = inputs.shape.as_list()
    mean, variance = tf.nn.moments(
        x=inputs,
        axes=[0] + list(range(2, len(shape))),
        keep_dims=True
    )
    moving_mean = tf.get_variable(
        name="moving_mean",
        shape=[1, shape[1]] + [1] * len(shape[2:]),
        initializer=tf.initializers.zeros(),
        trainable=False
    )
    moving_variance = tf.get_variable(
        name="moving_variance",
        shape=[1, shape[1]] + [1] * len(shape[2:]),
        initializer=tf.initializers.ones(),
        trainable=False
    )
    mean = tf.cond(training, lambda: mean, lambda: moving_mean)
    variance = tf.cond(training, lambda: variance, lambda: moving_variance)
    beta = tf.get_variable(
        name="beta",
        shape=[1, shape[1]] + [1] * len(shape[2:]),
        initializer=tf.initializers.zeros()
    )
    gamma = tf.get_variable(
        name="gamma",
        shape=[1, shape[1]] + [1] * len(shape[2:]),
        initializer=tf.initializers.ones()
    )
    inputs = tf.nn.batch_normalization(
        x=inputs,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )
    moving_mean = tf.cond(training, lambda: assign_moving_average(moving_mean, mean, momentum), lambda: moving_mean)
    moving_variance = tf.cond(training, lambda: assign_moving_average(moving_variance, variance, momentum), lambda: moving_variance)
    with tf.control_dependencies([moving_mean, moving_variance]):
        inputs = tf.indentity(inputs)
    return inputs


def group_normalization(inputs, groups, epsilon=1.0e-12):
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, groups, shape[1] // groups, *shape[2:]])
    mean, variance = tf.nn.moments(
        x=inputs,
        axes=list(range(2, len(shape) + 1)),
        keep_dims=True
    )
    beta = tf.get_variable(
        name="beta",
        shape=[1, groups, shape[1] // groups] + [1] * len(shape[2:]),
        initializer=tf.initializers.zeros()
    )
    gamma = tf.get_variable(
        name="gamma",
        shape=[1, groups, shape[1] // groups] + [1] * len(shape[2:]),
        initializer=tf.initializers.ones()
    )
    inputs = tf.nn.batch_normalization(
        x=inputs,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )
    inputs = tf.reshape(inputs, [-1, *shape[1:]])
    return inputs


def get_weight(shape,
               variance_scale=2.0,
               scale_weight=False,
               apply_weight_standardization=False,
               apply_spectral_normalization=False):
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
    if apply_spectral_normalization:
        weight = spectral_normalization(weight)
    return weight


def get_bias(shape):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=tf.initializers.zeros()
    )
    return bias


def dense(inputs,
          units,
          use_bias=True,
          variance_scale=2.0,
          scale_weight=False,
          apply_weight_standardization=False,
          apply_spectral_normalization=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization,
        apply_spectral_normalization=apply_spectral_normalization
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def embedding(inputs,
              units,
              variance_scale=2.0,
              scale_weight=False,
              apply_weight_standardization=False,
              apply_spectral_normalization=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization,
        apply_spectral_normalization=apply_spectral_normalization
    )
    inputs = tf.nn.embedding_lookup(weight, tf.argmax(inputs, axis=1))
    return inputs


def conv2d(inputs,
           filters,
           kernel_size,
           strides=[1, 1],
           use_bias=True,
           variance_scale=2.0,
           scale_weight=False,
           apply_weight_standardization=False,
           apply_spectral_normalization=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization,
        apply_spectral_normalization=apply_spectral_normalization
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


def conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     strides=[1, 1],
                     use_bias=True,
                     variance_scale=2.0,
                     scale_weight=False,
                     apply_weight_standardization=False,
                     apply_spectral_normalization=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_weight_standardization=apply_weight_standardization,
        apply_spectral_normalization=apply_spectral_normalization
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


def pixel_normalization(inputs, epsilon=1.0e-8):
    pixel_norm = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
    inputs = inputs / pixel_norm
    return inputs


def batch_stddev(inputs, groups=4, epsilon=1.0e-8):
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [groups, -1, *shape[1:]])
    inputs = tf.nn.moments(inputs, axes=[0])[1]
    inputs = tf.sqrt(inputs + epsilon)
    inputs = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
    inputs = tf.tile(inputs, [groups, 1, *shape[2:]])
    return inputs
