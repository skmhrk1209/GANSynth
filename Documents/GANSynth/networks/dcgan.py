#=================================================================================================#
# SNDCGAN Architecture
# [Spectral Normalization for Generative Adversarial Networks]
# (https://arxiv.org/pdf/1802.05957.pdf)
#
# based on Progressive Growing GAN Architecture
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
from . import pggan
from . import ops


class Generator(pggan.Generator):

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        with tf.variable_scope(name, reuse=None):

            resolution = self.min_resolution << index
            filters = self.max_filters >> index

            inputs = ops.dense(
                inputs=inputs,
                units=resolution * resolution * filters,
                name="dense_0"
            )

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="batch_normalization_0"
            )

            inputs = tf.nn.relu(inputs)

            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1, resolution, resolution, filters]
            )

            if self.data_format == "channels_first":

                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            return inputs

    def deconv2d_block(self, inputs, index, training, name="deconv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            filters = self.max_filters >> index

            inputs = ops.deconv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[4, 4],
                strides=[2, 2],
                data_format=self.data_format,
                name="deconv2d_0"
            )

            inputs = ops.batch_normalization(
                inputs=inputs,
                data_format=self.data_format,
                training=training,
                name="batch_normalization_0"
            )

            inputs = tf.nn.relu(inputs)

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = ops.deconv2d(
                inputs=inputs,
                filters=3,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                name="deconv2d_0"
            )

            inputs = tf.nn.sigmoid(inputs)

            return inputs


class Discriminator(pggan.Discriminator):

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = tf.layers.flatten(inputs)

            inputs = ops.dense(
                inputs=inputs,
                units=1,
                apply_spectral_normalization=True,
                name="dense_0"
            )

            return inputs

    def conv2d_block(self, inputs, index, training, name="conv2d_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            filters = self.max_filters >> (index - 1)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[4, 4],
                strides=[2, 2],
                data_format=self.data_format,
                apply_spectral_normalization=True,
                name="conv2d_0"
            )

            inputs = tf.nn.leaky_relu(inputs)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                apply_spectral_normalization=True,
                name="conv2d_1"
            )

            inputs = tf.nn.leaky_relu(inputs)

            return inputs

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            filters = self.max_filters >> (index - 1)

            inputs = ops.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                data_format=self.data_format,
                apply_spectral_normalization=True,
                name="conv2d_0"
            )

            inputs = tf.nn.leaky_relu(inputs)

            return inputs
