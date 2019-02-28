import tensorflow as tf
import numpy as np


def spectral_norm(inputs, singular_value="right", epsilon=1e-12):
    ''' Spectral Normalization
        [Spectral Normalization for Generative Adversarial Networks]
        (https://arxiv.org/pdf/1802.05957.pdf)
        this implementation is from google
        (https://github.com/google/compare_gan/blob/master/compare_gan/architectures/arch_ops.py)
    Args:
      inputs: The weight tensor to normalize.
      epsilon: Epsilon for L2 normalization.
      singular_value: Which first singular value to store (left or right). 
    Returns:
      The normalized weight tensor.
    '''

    with tf.variable_scope("spectral_norm"):

        # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
        # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
        # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
        # layers that put output channels as last dimension. This implies that w
        # here is equivalent to w.T in the paper.
        w = tf.reshape(inputs, [-1, inputs.shape[-1]])

        # Choose whether to persist the first left or first right singular vector.
        # As the underlying matrix is PSD, this should be equivalent, but in practice
        # the shape of the persisted vector is different. Here one can choose whether
        # to maintain the left or right one, or pick the one which has the smaller
        # dimension. We use the same variable for the singular vector if we switch
        # from normal weights to EMA weights.
        u_shape = [w.shape[0], 1] if singular_value == "left" else [1, w.shape[-1]]
        u_var = tf.get_variable(
            name="u_var",
            shape=u_shape,
            dtype=w.dtype,
            initializer=tf.random_normal_initializer(),
            trainable=False
        )
        u = u_var

        # Use power iteration method to approximate the spectral norm.
        # The authors suggest that one round of power iteration was sufficient in the
        # actual experiment to achieve satisfactory performance.
        power_iteration_rounds = 1
        for _ in range(power_iteration_rounds):
            if singular_value == "left":
                # `v` approximates the first right singular vector of matrix `w`.
                v = tf.nn.l2_normalize(tf.matmul(w, u, transpose_a=True), epsilon=epsilon)
                u = tf.nn.l2_normalize(tf.matmul(w, v), epsilon=epsilon)
            else:
                v = tf.nn.l2_normalize(tf.matmul(u, w, transpose_b=True), epsilon=epsilon)
                u = tf.nn.l2_normalize(tf.matmul(v, w), epsilon=epsilon)

        # Update the approximation.
        with tf.control_dependencies([tf.assign(u_var, u, name="update_u")]):
            u = tf.identity(u)

        # The authors of SN-GAN chose to stop gradient propagating through u and v
        # and we maintain that option.
        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        if singular_value == "left":
            norm_value = tf.matmul(tf.matmul(u, w, transpose_a=True), v)
        else:
            norm_value = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

        norm_value.shape.assert_is_fully_defined()
        norm_value.shape.assert_is_compatible_with([1, 1])

        w_normalized = w / norm_value

        # Deflate normalized weights to match the unnormalized tensor.
        w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)

    return w_tensor_normalized


def get_weight(shape, variance_scale=2, scale_weight=False, apply_spectral_norm=False):
    stddev = np.sqrt(variance_scale / np.prod(shape[:-1]))
    if scale_weight:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.initializers.random_normal(0.0, 1.0)
        ) * stddev
    else:
        weight = tf.get_variable(
            name="weight",
            shape=shape,
            initializer=tf.initializers.random_normal(0.0, stddev)
        )
    if apply_spectral_norm:
        weight = spectral_norm(weight)
    return weight


def get_bias(shape):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=tf.initializers.zeros()
    )
    return bias


def dense(inputs, units, use_bias=True, variance_scale=2, scale_weight=False, apply_spectral_norm=False):
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
           variance_scale=2, scale_weight=False, apply_spectral_norm=False):
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
                     variance_scale=2, scale_weight=False, apply_spectral_norm=False):
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
    inputs = tf.reshape(inputs, [group_size, -1, *shape[1:]])
    inputs -= tf.reduce_mean(inputs, axis=0, keepdims=True)
    inputs = tf.square(inputs)
    inputs = tf.reduce_mean(inputs, axis=0)
    inputs = tf.sqrt(inputs + epsilon)
    inputs = tf.reduce_mean(inputs, axis=[1, 2, 3], keepdims=True)
    inputs = tf.tile(inputs, [group_size, 1, *shape[2:]])
    return inputs


def embedding(inputs, units, variance_scale=2, scale_weight=False, apply_spectral_norm=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        variance_scale=variance_scale,
        scale_weight=scale_weight,
        apply_spectral_norm=apply_spectral_norm
    )
    inputs = tf.nn.embedding_lookup(weight, tf.argmax(inputs, axis=1))
    return inputs


def scale(inputs, in_min, in_max, out_min, out_max):
    inputs = out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)
    return inputs
