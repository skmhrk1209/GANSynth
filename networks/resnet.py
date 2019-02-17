#=================================================================================================#
# ResNet Architecture
# [Deep Residual Learning for Image Recognition]
# (https://arxiv.org/pdf/1512.03385.pdf)
#
# based on Progressive Growing GAN Architecture
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
from . import pggan
from . import ops


class Generator(pggan.Generator):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        with tf.variable_scope(name, reuse=None):

            resolution = self.min_resolution << index
            filters = self.max_filters >> index

            inputs = ops.dense(
                inputs=inputs,
                units=resolution * resolution * filters,
                use_bias=False,
                name="dense_0"
            )

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1, resolution, resolution, filters]
            )

            return inputs

    def deconv2d_block(self, inputs, index, training, name="deconv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            filters = self.max_filters >> index

            inputs = ops.upsampling2d(
                inputs=inputs,
                factors=[2, 2],
                data_format=self.data_format,
                dynamic=False
            )

            inputs = ops.residual_block(
                inputs=inputs,
                filters=filters,
                strides=[1, 1],
                use_bias=False,
                normalization=ops.batch_normalization,
                activation=tf.nn.relu,
                data_format=self.data_format,
                training=training,
                name="residual_block_0"
            )

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="batch_normalization_0"
            )

            inputs = tf.nn.relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[3, 3],
                strides=[1, 1],
                use_bias=True,
                data_format=self.data_format,
                name="conv2d_0"
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(pggan.Discriminator):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def dense_block(self, inputs, conditions, index, training, name="dense_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = tf.nn.relu(inputs)

            inputs = ops.global_average_pooling2d(
                inputs=inputs,
                data_format=self.data_format
            )

            logits = ops.dense(
                inputs=inputs,
                units=1,
                use_bias=True,
                apply_spectral_normalization=True,
                name="logits"
            )

            embedded = ops.dense(
                inputs=conditions,
                units=inputs.shape[-1],
                use_bias=False,
                apply_spectral_normalization=True,
                name="embedded"
            )

            logits += tf.reduce_sum(
                input_tensor=inputs * embedded,
                keepdims=True
            )

            return logits

    def conv2d_block(self, inputs, index, training, name="conv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            filters = self.max_filters >> (index - 1)

            inputs = ops.residual_block(
                inputs=inputs,
                filters=filters,
                strides=[1, 1],
                data_format=self.data_format,
                use_bias=True,
                apply_spectral_normalization=True,
                normalization=None,
                training=training,
                activation=tf.nn.relu,
                name="residual_block_0"
            )

            inputs = ops.downsampling2d(
                inputs=inputs,
                factors=[2, 2],
                data_format=self.data_format
            )

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            filters = self.max_filters >> (index - 1)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                use_bias=True,
                data_format=self.data_format,
                apply_spectral_normalization=True,
                name="conv2d_0"
            )

            inputs = tf.nn.leaky_relu(inputs)

            return inputs
