import tensorflow as tf
import numpy as np
from ops import *


class PGGAN(object):

    def __init__(self, min_resolutions, max_resolutions, min_filters, max_filters, num_channels, apply_spectral_norm):

        self.min_resolutions = min_resolutions
        self.max_resolutions = max_resolutions
        self.min_resolution = max(min_resolutions)
        self.max_resolution = max(max_resolutions)
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.num_channels = num_channels
        self.apply_spectral_norm = apply_spectral_norm

    def generator(self, latents, labels, target_resolution, name="ganerator", reuse=None):

        def conv_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(resolution), reuse=reuse):
                if resolution == self.min_resolution:
                    inputs = pixel_norm(inputs)
                    with tf.variable_scope("dense"):
                        inputs = dense(
                            inputs=inputs,
                            units=self.max_filters * np.prod(self.min_resolutions)
                        )
                        inputs = tf.reshape(inputs, [-1, self.max_filters, *self.min_resolutions])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=self.max_filters,
                            kernel_size=[3, 3]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                else:
                    with tf.variable_scope("conv_upscale"):
                        inputs = conv2d_transpose(
                            inputs=inputs,
                            filters=inputs.shape[1].value >> 1,
                            kernel_size=[3, 3],
                            strides=[2, 2]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=inputs.shape[1].value,
                            kernel_size=[3, 3]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                return inputs

        def color_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(resolution), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=self.num_channels,
                        kernel_size=[1, 1],
                        variance_scale=1
                    )
                    inputs = tf.nn.tanh(inputs)
                return inputs

        def lerp(a, b, t): return (1 - t) * a + t * b

        def grow(feature_maps, resolution):

            def high_resolution_images():
                return grow(conv_block(feature_maps, resolution), resolution << 1)

            def middle_resolution_images():
                return upscale2d(color_block(conv_block(feature_maps, resolution), resolution), [self.max_resolution // resolution] * 2)

            def low_resolution_images():
                return upscale2d(color_block(feature_maps, resolution >> 1), [self.max_resolution // (resolution >> 1)] * 2)

            predicate = tf.greater(target_resolution, resolution)
            lerp_coefficient = tf.cast((target_resolution - (resolution >> 1)) / (resolution - (resolution >> 1)), tf.float32)

            if resolution == self.min_resolution:

                images = tf.cond(
                    pred=predicate,
                    true_fn=high_resolution_images,
                    false_fn=middle_resolution_images
                )

            elif resolution == self.max_resolution:

                images = tf.cond(
                    pred=predicate,
                    true_fn=lambda: color_block(conv_block(feature_maps, resolution), resolution),
                    false_fn=lambda: lerp(
                        a=upscale2d(color_block(feature_maps, resolution >> 1)),
                        b=color_block(conv_block(feature_maps, resolution), resolution),
                        t=lerp_coefficient
                    )
                )

            else:

                images = tf.cond(
                    pred=predicate,
                    true_fn=high_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=lerp_coefficient
                    )
                )

            return images

        with tf.variable_scope(name, reuse=reuse):

            return grow(tf.concat([latents, labels], axis=-1), self.min_resolution)

    def discriminator(self, images, labels, target_resolution, name="dicriminator", reuse=None):

        def color_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(resolution), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=self.min_filters * (self.max_resolution // resolution),
                        kernel_size=[1, 1]
                    )
                    inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def conv_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(resolution), reuse=reuse):
                if resolution > self.min_resolution:
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=inputs.shape[1].value,
                            kernel_size=[3, 3],
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("conv_downscale"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=inputs.shape[1].value << 1,
                            kernel_size=[3, 3],
                            strides=[2, 2],
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                else:
                    inputs = batch_stddev(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=inputs.shape[1].value,
                            kernel_size=[3, 3],
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("global_average_pooling"):
                        inputs = global_average_pooling2d(inputs)
                    with tf.variable_scope("logits"):
                        logits = dense(
                            inputs=inputs,
                            units=1,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                    with tf.variable_scope("projection"):
                        inputs = logits + projection(
                            inputs=inputs,
                            labels=labels,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                return inputs

        def lerp(a, b, t): return (1 - t) * a + t * b

        def grow(images, resolution):

            def high_resolution_feature_maps():
                return conv_block(grow(images, resolution << 1), resolution)

            def middle_resolution_feature_maps():
                return conv_block(color_block(downscale2d(images, [self.max_resolution // resolution] * 2), resolution), resolution)

            def low_resolution_feature_maps():
                return color_block(downscale2d(images, [self.max_resolution // (resolution >> 1)] * 2), resolution >> 1)

            predicate = tf.greater(target_resolution, resolution)
            lerp_coefficient = tf.cast((target_resolution - (resolution >> 1)) / (resolution - (resolution >> 1)), tf.float32)

            if resolution == self.min_resolution:

                feature_maps = tf.cond(
                    pred=predicate,
                    true_fn=high_resolution_feature_maps,
                    false_fn=middle_resolution_feature_maps
                )

            elif resolution == self.max_resolution:

                feature_maps = tf.cond(
                    pred=predicate,
                    true_fn=lambda: conv_block(color_block(images, resolution), resolution),
                    false_fn=lambda: lerp(
                        a=color_block(downscale2d(images), resolution >> 1),
                        b=conv_block(color_block(images, resolution), resolution),
                        t=lerp_coefficient
                    )
                )

            else:

                feature_maps = tf.cond(
                    pred=predicate,
                    true_fn=high_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=lerp_coefficient
                    )
                )

            return feature_maps

        with tf.variable_scope(name, reuse=reuse):

            return grow(images, self.min_resolution)
