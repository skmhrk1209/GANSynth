import numpy as np
import tensorflow as tf
from ops import *


class PGGAN(object):

    def __init__(self, min_resolution, max_resolution, min_filters, max_filters, num_channels):

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.num_channels = num_channels

    def generator(self, latents, labels, target_resolution, name="ganerator", reuse=None):

        def conv_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(resolution), reuse=reuse):
                if resolution == self.min_resolution:
                    inputs = pixel_normalization(inputs)
                    with tf.variable_scope("dense"):
                        inputs = dense(inputs, units=self.max_filters * self.min_resolution * self.min_resolution)
                        inputs = tf.reshape(inputs, [-1, self.max_filters, self.min_resolution, self.min_resolution])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_normalization(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(inputs, filters=self.max_filters, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_normalization(inputs)
                else:
                    with tf.variable_scope("conv_upscale"):
                        inputs = conv2d_transpose(inputs, filters=inputs.shape[1].value >> 1, kernel_size=[3, 3], strides=[2, 2])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_normalization(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(inputs, filters=inputs.shape[1].value, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_normalization(inputs)
                return inputs

        def color_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(resolution), reuse=reuse):
                inputs = conv2d(inputs, filters=self.num_channels, kernel_size=[1, 1], variance_scale=1)
                inputs = tf.nn.tanh(inputs)
                return inputs

        def lerp(a, b, t): return (1 - t) * a + t * b

        def grow(inputs, resolution):

            if resolution == self.min_resolution:

                feature_maps = conv_block(inputs, resolution)

                images = tf.cond(
                    pred=tf.greater(target_resolution, resolution),
                    true_fn=lambda: grow(feature_maps, resolution << 1),
                    false_fn=lambda: upscale2d(color_block(feature_maps, resolution), [self.max_resolution // resolution] * 2)
                )

            elif resolution == self.max_resolution:

                feature_maps = conv_block(inputs, resolution)

                images = lerp(
                    a=upscale2d(color_block(inputs, resolution >> 1)),
                    b=color_block(feature_maps, resolution),
                    t=tf.cast((target_resolution - (resolution >> 1)) / (resolution - (resolution >> 1)), tf.float32)
                )

            else:

                feature_maps = conv_block(inputs, resolution)

                images = tf.cond(
                    pred=tf.greater(target_resolution, resolution),
                    true_fn=lambda: grow(feature_maps, resolution << 1),
                    false_fn=lambda: upscale2d(lerp(
                        a=upscale2d(color_block(inputs, resolution >> 1)),
                        b=color_block(feature_maps, resolution),
                        t=tf.cast((target_resolution - (resolution >> 1)) / (resolution - (resolution >> 1)), tf.float32)
                    ), [self.max_resolution // resolution] * 2)
                )

            return images

        with tf.variable_scope(name, reuse=reuse):

            return grow(tf.concat([latents, labels], axis=-1), self.min_resolution)

    def discriminator(self, images, labels, target_resolution, name="dicriminator", reuse=None):

        def color_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(resolution), reuse=reuse):
                inputs = conv2d(inputs, filters=self.min_filters * (self.max_resolution // resolution), kernel_size=[1, 1])
                inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def conv_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(resolution), reuse=reuse):
                if resolution > self.min_resolution:
                    with tf.variable_scope("conv"):
                        inputs = conv2d(inputs, filters=inputs.shape[1].value, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("conv_downscale"):
                        inputs = conv2d(inputs, filters=inputs.shape[1].value << 1, kernel_size=[3, 3], strides=[2, 2])
                        inputs = tf.nn.leaky_relu(inputs)
                else:
                    inputs = batch_stddev(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(inputs, filters=inputs.shape[1].value, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("logits"):
                        features = global_average_pooling2d(inputs)
                        inputs = dense(features, units=1, variance_scale=1)
                    with tf.variable_scope("projection"):
                        inputs += projection(features, labels)
                return inputs

        def lerp(a, b, t): return (1 - t) * a + t * b

        def grow(inputs, resolution):

            if resolution == self.min_resolution:

                feature_maps = tf.cond(
                    pred=tf.greater(target_resolution, resolution),
                    true_fn=lambda: grow(inputs, resolution << 1),
                    false_fn=lambda: color_block(downscale2d(inputs, [self.max_resolution // resolution] * 2), resolution)
                )

                feature_maps = conv_block(feature_maps, resolution)

            elif resolution == self.max_resolution:

                feature_maps = conv_block(color_block(inputs, resolution), resolution)

                feature_maps = tf.cond(
                    pred=tf.greater(target_resolution, resolution),
                    true_fn=lambda: feature_maps,
                    false_fn=lambda: lerp(
                        a=color_block(downscale2d(inputs, [self.max_resolution // (resolution >> 1)] * 2), resolution >> 1),
                        b=feature_maps,
                        t=tf.cast((target_resolution - (resolution >> 1)) / (resolution - (resolution >> 1)), tf.float32)
                    )
                )

            else:

                feature_maps = tf.cond(
                    pred=tf.greater(target_resolution, resolution),
                    true_fn=lambda: grow(inputs, resolution << 1),
                    false_fn=lambda: color_block(downscale2d(inputs, [self.max_resolution // resolution] * 2), resolution)
                )

                feature_maps = conv_block(feature_maps, resolution)

                feature_maps = tf.cond(
                    pred=tf.greater(target_resolution, resolution),
                    true_fn=lambda: feature_maps,
                    false_fn=lambda: lerp(
                        a=color_block(downscale2d(inputs, [self.max_resolution // (resolution >> 1)] * 2), resolution >> 1),
                        b=feature_maps,
                        t=tf.cast((target_resolution - (resolution >> 1)) / (resolution - (resolution >> 1)), tf.float32)
                    )
                )

            return feature_maps

        with tf.variable_scope(name, reuse=reuse):

            return grow(images, self.min_resolution)
