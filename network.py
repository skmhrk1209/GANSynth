import numpy as np
import tensorflow as tf
import ops


class PGGAN(object):

    def __init__(self, resolution, min_resolution=4, max_resolution=256, min_filters=4, max_filters=256, num_channels=3, num_classes=10):

        self.resolution = resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.num_channels = num_channels
        self.num_classes = num_classes

    def generator(self, inputs, name="ganerator", reuse=None):

        def conv_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(resolution), reuse=reuse):
                if resolution == self.min_resolution:
                    inputs = ops.pixel_normalization(inputs)
                    with tf.variable_scope("dense"):
                        inputs = ops.dense(inputs, units=self.max_filters * self.min_resolution * self.min_resolution)
                        inputs = tf.reshape(inputs, [-1, self.max_filters, self.min_resolution, self.min_resolution])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = ops.pixel_normalization(inputs)
                    with tf.variable_scope("conv"):
                        inputs = ops.conv2d(inputs, filters=self.max_filters, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = ops.pixel_normalization(inputs)
                else:
                    with tf.variable_scope("conv_upscale"):
                        inputs = ops.conv2d_transpose(inputs, filters=inputs.shape[1].value >> 1, kernel_size=[3, 3], strides=[2, 2])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = ops.pixel_normalization(inputs)
                    with tf.variable_scope("conv"):
                        inputs = ops.conv2d(inputs, filters=inputs.shape[1].value, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = ops.pixel_normalization(inputs)
                return inputs

        def color_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(resolution), reuse=reuse):
                inputs = ops.conv2d(inputs, filters=self.num_channels, kernel_size=[1, 1], variance_scale=1)
                return inputs

        def lerp(a, b, t): return (1 - t) * a + t * b

        def grow(inputs, resolution):

            if resolution == self.min_resolution:

                outputs = conv_block(inputs, resolution)

                outputs = tf.cond(
                    pred=tf.greater(self.resolution, resolution),
                    true_fn=lambda: grow(outputs, resolution << 1),
                    false_fn=lambda: color_block(outputs, resolution)
                )

            elif resolution == self.max_resolution:

                outputs = lerp(
                    a=ops.upscale2d(color_block(inputs, resolution >> 1)),
                    b=color_block(conv_block(inputs, resolution), resolution),
                    t=(self.resolution - (resolution >> 1)) / (resolution - (resolution >> 1))
                )

            else:

                outputs = conv_block(inputs, resolution)

                outputs = tf.cond(
                    pred=tf.greater(self.resolution, resolution),
                    true_fn=lambda: grow(outputs, resolution << 1),
                    false_fn=lambda: color_block(outputs, resolution)
                )

                outputs = tf.cond(
                    pred=tf.greater(self.resolution, resolution),
                    true_fn=lambda: outputs,
                    false_fn=lambda: ops.upscale2d(lerp(
                        a=ops.upscale2d(color_block(inputs, resolution >> 1)),
                        b=outputs,
                        t=(self.resolution - (resolution >> 1)) / (resolution - (resolution >> 1))
                    ), [self.max_resolution // resolution] * 2)
                )

            return outputs

        with tf.variable_scope(name, reuse=reuse):

            return grow(inputs, self.min_resolution)

    def discriminator(self, inputs, name="dicriminator", reuse=None):

        def color_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{0}x{0}".format(resolution), reuse=reuse):
                inputs = ops.conv2d(inputs, filters=self.min_filters * (self.max_resolution // resolution), kernel_size=[1, 1])
                inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def conv_block(inputs, resolution, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv_block_{0}x{0}".format(resolution), reuse=reuse):
                if resolution > self.min_resolution:
                    with tf.variable_scope("conv"):
                        inputs = ops.conv2d(inputs, filters=inputs.shape[1].value, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("conv_downscale"):
                        inputs = ops.conv2d(inputs, filters=inputs.shape[1].value << 1, kernel_size=[3, 3], strides=[2, 2])
                        inputs = tf.nn.leaky_relu(inputs)
                else:
                    inputs = ops.batch_stddev(inputs)
                    with tf.variable_scope("conv"):
                        inputs = ops.conv2d(inputs, filters=inputs.shape[1].value, kernel_size=[3, 3])
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("dense"):
                        inputs = tf.layers.flatten(inputs)
                        inputs = ops.dense(inputs, units=inputs.shape[1].value << 1)
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("logits"):
                        inputs = ops.dense(inputs, units=self.num_classes + 1, variance_scale=1)
                return inputs

        def lerp(a, b, t): return (1 - t) * a + t * b

        def grow(inputs, resolution):

            if resolution == self.min_resolution:

                outputs = tf.cond(
                    pred=tf.greater(self.resolution, resolution),
                    true_fn=lambda: grow(inputs, resolution << 1),
                    false_fn=lambda: color_block(ops.downscale2d(inputs, [self.max_resolution // resolution] * 2), resolution)
                )

                outputs = conv_block(outputs, resolution)

            elif resolution == self.max_resolution:

                outputs = lerp(
                    a=color_block(ops.downscale2d(inputs, [self.max_resolution // (resolution >> 1)] * 2), resolution >> 1),
                    b=conv_block(color_block(inputs, resolution), resolution),
                    t=(self.resolution - (resolution >> 1)) / (resolution - (resolution >> 1))
                )

            else:

                outputs = tf.cond(
                    pred=tf.greater(self.resolution, resolution),
                    true_fn=lambda: grow(inputs, resolution << 1),
                    false_fn=lambda: color_block(ops.downscale2d(inputs, [self.max_resolution // resolution] * 2), resolution)
                )

                outputs = conv_block(outputs, resolution)

                outputs = tf.cond(
                    pred=tf.greater(self.resolution, resolution),
                    true_fn=lambda: outputs,
                    false_fn=lambda: lerp(
                        a=color_block(ops.downscale2d(inputs, [self.max_resolution // (resolution >> 1)] * 2), resolution >> 1),
                        b=outputs,
                        t=(self.resolution - (resolution >> 1)) / (resolution - (resolution >> 1))
                    )
                )

            return outputs

        with tf.variable_scope(name, reuse=reuse):

            return grow(inputs, self.min_resolution)
