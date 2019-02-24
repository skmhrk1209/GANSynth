import tensorflow as tf
import numpy as np
from ops import *


def log2(a, b):
    n = 0
    while (a != b).any():
        a <<= 1
        n += 1
    return n


def lerp(a, b, t):
    return t * a + (1 - t) * b


class PGGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels, apply_spectral_norm):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.apply_spectral_norm = apply_spectral_norm

        self.min_depth = log2(self.min_resolution, self.min_resolution)
        self.max_depth = log2(self.min_resolution, self.max_resolution)

    def generator(self, latents, labels, progress, name="ganerator", reuse=None):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    inputs = pixel_norm(inputs)
                    with tf.variable_scope("dense"):
                        inputs = dense(
                            inputs=inputs,
                            units=channels(depth) * resolution(depth).prod()
                        )
                        inputs = tf.reshape(inputs, [-1, channels(depth), *resolution(depth)])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                else:
                    with tf.variable_scope("conv_upscale"):
                        inputs = conv2d_transpose(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            strides=[2, 2]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(*resolution(depth)), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=2,
                        kernel_size=[1, 1],
                        variance_scale=1
                    )
                    inputs = tf.nn.tanh(inputs)
                return inputs

        out_depth = scale(progress, 0.0, 1.0, self.min_depth, self.max_depth)

        def grow(feature_maps, depth):

            def high_resolution_images():
                return grow(conv_block(feature_maps, depth), depth + 1)

            def middle_resolution_images():
                return upscale2d(color_block(conv_block(feature_maps, depth), depth), [1 << (self.max_depth - depth)] * 2)

            def low_resolution_images():
                return upscale2d(color_block(feature_maps, depth - 1), [1 << (self.max_depth - (depth - 1))] * 2)

            if depth == self.min_depth:
                images = tf.cond(
                    pred=tf.greater(out_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=middle_resolution_images
                )
            elif depth == self.max_depth:
                images = tf.cond(
                    pred=tf.greater(out_depth, depth),
                    true_fn=middle_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - out_depth
                    )
                )
            else:
                images = tf.cond(
                    pred=tf.greater(out_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - out_depth
                    )
                )
            return images

        with tf.variable_scope(name, reuse=reuse):
            return grow(tf.concat([latents, labels], axis=-1), self.min_depth)

    def discriminator(self, images, labels, progress, name="dicriminator", reuse=None):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    inputs = batch_stddev(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
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
                else:
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("conv_downscale"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth - 1),
                            kernel_size=[3, 3],
                            strides=[2, 2],
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(*resolution(depth)), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=channels(depth),
                        kernel_size=[1, 1]
                    )
                    inputs = tf.nn.leaky_relu(inputs)
                return inputs

        in_depth = scale(progress, 0.0, 1.0, self.min_depth, self.max_depth)

        def grow(images, depth):

            def high_resolution_feature_maps():
                return conv_block(grow(images, depth + 1), depth)

            def middle_resolution_feature_maps():
                return conv_block(color_block(downscale2d(images, [1 << (self.max_depth - depth)] * 2), depth), depth)

            def low_resolution_feature_maps():
                return color_block(downscale2d(images, [1 << (self.max_depth - (depth - 1))] * 2), depth - 1)

            if depth == self.min_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(in_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=middle_resolution_feature_maps
                )
            elif depth == self.max_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(in_depth, depth),
                    true_fn=middle_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=depth - in_depth
                    )
                )
            else:
                feature_maps = tf.cond(
                    pred=tf.greater(in_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=depth - in_depth
                    )
                )
            return feature_maps

        with tf.variable_scope(name, reuse=reuse):
            return grow(images, self.min_depth)
