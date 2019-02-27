import tensorflow as tf
import numpy as np


def spectral_norm(input):
    ''' spectral normalization
        [Spectral Normalization for Generative Adversarial Networks]
        (https://arxiv.org/pdf/1802.05957.pdf)
        this implementation is from google
        (https://github.com/google/compare_gan/blob/master/compare_gan/architectures/arch_ops.py)
    '''

    if len(input.shape) < 2:
        raise ValueError("Spectral norm can only be applied to multi-dimensional tensors")

    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). But Sonnet's and Compare_gan's Conv2D kernel
    # weight shape is (KH, KW, C_in, C_out), so it should be reshaped to
    # (KH * KW * C_in, C_out), and similarly for other layers that put output
    # channels as last dimension.
    # n.b. this means that w here is equivalent to w.T in the paper.
    w = tf.reshape(input, [-1, input.shape[-1]])

    # Persisted approximation of first left singular vector of matrix `w`.

    u_var = tf.get_variable(
        name="u_var",
        shape=[w.shape[0], 1],
        initializer=tf.random_normal_initializer(),
        trainable=False
    )
    u = u_var

    # Use power iteration method to approximate spectral norm.
    # The authors suggest that "one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance". According to
    # observation, the spectral norm become very accurate after ~20 steps.

    power_iteration_rounds = 1
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.nn.l2_normalize(tf.matmul(tf.transpose(w), u))
        u = tf.nn.l2_normalize(tf.matmul(w, v))

    # Update persisted approximation.
    with tf.control_dependencies([tf.assign(u_var, u)]):
        u = tf.identity(u)

    # The authors of SN-GAN chose to stop gradient propagating through u and v.
    # In johnme@'s experiments it wasn't clear that this helps, but it doesn't
    # seem to hinder either so it's kept in order to be a faithful implementation.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    # Largest singular value of `w`.
    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])

    w_normalized = w / norm_value

    # Unflatten normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, input.shape)
    return w_tensor_normalized


def get_weight(shape, variance_scale=2, scale_weight=True, apply_spectral_norm=False):
    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
    if scale_weight:
        weight = tf.get_variable("weight", shape=shape, initializer=tf.initializers.random_normal()) * stddev
    else:
        weight = tf.get_variable("weight", shape=shape, initializer=tf.initializers.random_normal(0, stddev))
    if apply_spectral_norm:
        weight = spectral_norm(weight)
    return weight


def get_bias(shape):
    bias = tf.get_variable("bias", shape=shape, initializer=tf.initializers.zeros())
    return bias


def dense(inputs, units, use_bias=True, variance_scale=2, scale_weight=True, apply_spectral_norm=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_spectral_norm=apply_spectral_norm
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias([inputs.shape[1].value])
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d(inputs, filters, kernel_size, strides=[1, 1], use_bias=True,
           variance_scale=2, scale_weight=True, apply_spectral_norm=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_spectral_norm=apply_spectral_norm
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
                     variance_scale=2, scale_weight=True, apply_spectral_norm=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_spectral_norm=apply_spectral_norm
    )
    weight = tf.transpose(weight, [0, 1, 3, 2])
    input_shape = np.array(inputs.shape.as_list())
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
    shape = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
    inputs = tf.tile(inputs, [1, 1, 1, factors[0], 1, factors[1]])
    inputs = tf.reshape(inputs, [-1, shape[1], shape[2] * factors[0], shape[3] * factors[1]])
    return inputs


def downscale2d(inputs, factors=[2, 2]):
    # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True
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


def pixel_norm(inputs, epsilon=1e-8):
    inputs *= tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + epsilon)
    return inputs


def batch_stddev(inputs, group_size=4, epsilon=1e-8):
    shape = inputs.shape.as_list()
    stddev = tf.reshape(inputs, [group_size, -1, *shape[1:]])
    stddev -= tf.reduce_mean(stddev, axis=0, keepdims=True)
    stddev = tf.square(stddev)
    stddev = tf.reduce_mean(stddev, axis=0)
    stddev = tf.sqrt(stddev + epsilon)
    stddev = tf.reduce_mean(stddev, axis=[1, 2, 3], keepdims=True)
    stddev = tf.tile(stddev, [group_size, 1, *shape[2:]])
    inputs = tf.concat([inputs, stddev], axis=1)
    return inputs


def global_average_pooling2d(inputs):
    inputs = tf.reduce_mean(inputs, axis=[2, 3])
    return inputs


def projection(inputs, labels, variance_scale=2, scale_weight=True, apply_spectral_norm=False):
    weight = get_weight(
        shape=[labels.shape[1].value, inputs.shape[1].value],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_spectral_norm=apply_spectral_norm
    )
    labels = tf.nn.embedding_lookup(weight, tf.argmax(labels, axis=1))
    inputs = tf.reduce_mean(inputs * labels, axis=1, keepdims=True)
    return inputs


def scale(inputs, in_min, in_max, out_min, out_max):
    inputs = out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)
    return inputs
