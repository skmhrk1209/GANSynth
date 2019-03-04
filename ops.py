import tensorflow as tf
import numpy as np


def get_weight(shape, initializer, apply_spectral_norm=False):
    weight = tf.get_variable(
        name="weight",
        shape=shape,
        initializer=initializer
    )
    if apply_spectral_norm:
        weight = spectral_norm(weight)
    return weight


def get_bias(shape, initializer):
    bias = tf.get_variable(
        name="bias",
        shape=shape,
        initializer=initializer
    )
    return bias


def dense(inputs, units, use_bias=True,
          weight_initializer=tf.initializers.glorot_normal(),
          bias_initializer=tf.initializers.zeros(),
          apply_spectral_norm=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        initializer=weight_initializer,
        apply_spectral_norm=apply_spectral_norm
    )
    inputs = tf.matmul(inputs, weight)
    if use_bias:
        bias = get_bias(
            shape=[inputs.shape[1].value],
            initializer=bias_initializer
        )
        inputs = tf.nn.bias_add(inputs, bias)
    return inputs


def conv2d(inputs, filters, kernel_size, strides=[1, 1], use_bias=True,
           weight_initializer=tf.initializers.he_normal(),
           bias_initializer=tf.initializers.zeros(),
           apply_spectral_norm=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        initializer=weight_initializer,
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
        bias = get_bias(
            shape=[inputs.shape[1].value],
            initializer=bias_initializer
        )
        inputs = tf.nn.bias_add(inputs, bias, data_format="NCHW")
    return inputs


def conv2d_transpose(inputs, filters, kernel_size, strides=[1, 1], use_bias=True,
                     weight_initializer=tf.initializers.he_normal(),
                     bias_initializer=tf.initializers.zeros(),
                     apply_spectral_norm=False):
    weight = get_weight(
        shape=[*kernel_size, inputs.shape[1].value, filters],
        weight_initializer=weight_initializer,
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
        bias = get_bias(
            shape=[inputs.shape[1].value],
            initializer=bias_initializer
        )
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


def embed(inputs, units,
          weight_initializer=tf.initializers.glorot_normal(),
          apply_spectral_norm=False):
    weight = get_weight(
        shape=[inputs.shape[1].value, units],
        initializer=weight_initializer,
        apply_spectral_norm=apply_spectral_norm
    )
    inputs = tf.nn.embedding_lookup(weight, tf.argmax(inputs, axis=1))
    return inputs


def conditional_batch_norm(inputs, labels, training, center=True, scale=True,
                           center_weight_initializer=tf.initializers.zeros(),
                           scale_weight_initializer=tf.initializers.ones(),
                           apply_spectral_norm=False):
    ''' Conditional Batch Normalization
        [Modulating early visual processing by language]
        (https://arxiv.org/pdf/1707.00683.pdf)
    '''
    # NOTE: fused version doesn't work
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        center=False,
        scale=False,
        training=training,
        fused=False
    )

    if scale:
        with tf.variable_scope("scale"):
            gamma = embed(
                inputs=labels,
                units=inputs.shape[1],
                weight_initializer=scale_weight_initializer,
                apply_spectral_norm=apply_spectral_norm
            )
            gamma = tf.reshape(
                tensor=gamma,
                shape=[-1, gamma.shape[1], 1, 1]
            )
        inputs *= gamma

    if center:
        with tf.variable_scope("center"):
            beta = embed(
                inputs=labels,
                units=inputs.shape[1],
                weight_initializer=center_weight_initializer,
                apply_spectral_norm=apply_spectral_norm
            )
            beta = tf.reshape(
                tensor=beta,
                shape=[-1, beta.shape[1], 1, 1]
            )
        inputs += beta

    return inputs


def adaptive_instance_norm(inputs, latents, use_bias=True, center=True, scale=True,
                           center_weight_initializer=tf.initializers.glorot_normal(),
                           center_bias_initializer=tf.initializers.zeros(),
                           scale_weight_initializer=tf.initializers.glorot_normal(),
                           scale_bias_initializer=tf.initializers.zeros(),
                           apply_spectral_norm=False,
                           epsilon=1e-8):
    ''' Adaptive Instance Normalization
        [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization]
        (https://arxiv.org/pdf/1703.06868.pdf)
    '''
    inputs -= tf.reduce_mean(inputs, axis=[2, 3], keepdims=True)
    inputs *= tf.rsqrt(tf.reduce_mean(tf.square(inputs), axis=[2, 3], keepdims=True) + epsilon)

    if scale:
        with tf.variable_scope("scale"):
            gamma = dense(
                inputs=latents,
                units=inputs.shape[1],
                use_bias=use_bias,
                weight_initializer=scale_weight_initializer,
                bias_initializer=scale_bias_initializer,
                apply_spectral_norm=apply_spectral_norm
            )
            gamma = tf.reshape(
                tensor=gamma,
                shape=[-1, gamma.shape[1], 1, 1]
            )
        inputs *= gamma

    if center:
        with tf.variable_scope("center"):
            beta = dense(
                inputs=latents,
                units=inputs.shape[1],
                use_bias=use_bias,
                weight_initializer=center_weight_initializer,
                bias_initializer=center_bias_initializer,
                apply_spectral_norm=apply_spectral_norm
            )
            beta = tf.reshape(
                tensor=beta,
                shape=[-1, beta.shape[1], 1, 1]
            )
        inputs += beta

    return inputs


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


def self_attention(inputs, filters,
                   weight_initializer=tf.initializers.glorot_normal(),
                   apply_spectral_norm=False):
    ''' Self Attention Mechanism
        [Self-Attention Generative Adversarial Networks]
        (https://arxiv.org/pdf/1805.08318.pdf)
    '''
    with tf.variable_scope("query"):
        queries = conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[1, 1],
            use_bias=False,
            weight_initializer=weight_initializer,
            apply_spectral_norm=apply_spectral_norm
        )
        queries = tf.reshape(
            tensor=queries,
            shape=[-1, queries.shape[1], np.prod(queries.shape[2:])]
        )
    with tf.variable_scope("key"):
        keys = conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[1, 1],
            use_bias=False,
            weight_initializer=weight_initializer,
            apply_spectral_norm=apply_spectral_norm
        )
        keys = tf.reshape(
            tensor=keys,
            shape=[-1, keys.shape[1], np.prod(keys.shape[2:])]
        )
    with tf.variable_scope("value"):
        values = conv2d(
            inputs=inputs,
            filters=inputs.shape[1],
            kernel_size=[1, 1],
            use_bias=False,
            weight_initializer=weight_initializer,
            apply_spectral_norm=apply_spectral_norm
        )
        values = tf.reshape(
            tensor=values,
            shape=[-1, values.shape[1], np.prod(values.shape[2:])]
        )
    attention_map = tf.matmul(
        a=queries,
        b=keys,
        transpose_a=True,
        transpose_b=False
    )
    attention_map = tf.nn.softmax(
        logits=attention_map,
        axis=-1
    )
    self_attention_maps = tf.matmul(
        a=values,
        b=attention_map,
        transpose_a=False,
        transpose_b=True
    )
    self_attention_maps = tf.reshape(
        tensor=self_attention_maps,
        shape=[-1, *inputs.shape[1:]]
    )
    gamma = tf.get_variable(
        name="gamma",
        shape=[],
        initializer=tf.initializers.zeros()
    )
    inputs += gamma * self_attention_maps
    return inputs
