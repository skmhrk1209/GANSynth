#=================================================================================================#
# Progressive Growing GAN Architecture
#
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
from . import ops


def lerp(a, b, t):
    return a + (b - a) * t


class Generator(object):

    def __init__(self, min_resolution, max_resolution, min_filters, max_filters, data_format):

        if (max_resolution // min_resolution) != (max_filters // min_filters):
            raise ValueError("Invalid number of filters")

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.data_format = data_format
        self.num_layers = int(np.log2(max_resolution // min_resolution)) + 2

    def __call__(self, inputs, coloring_index, training, name="generator", reuse=None):

        ceiled_coloring_index = tf.cast(tf.ceil(coloring_index), tf.int32)

        with tf.variable_scope(name, reuse=reuse):

            #========================================================================#
            # very complicated but efficient architecture
            # each layer has two output paths: feature_maps and images
            # whether which path is evaluated at runtime
            # depends on variable "coloring_index"
            # but, all possible pathes must be constructed at compile time
            #========================================================================#
            def grow(inputs, index):

                with tf.variable_scope("layer_{}".format(index)):

                    if index == 0:

                        feature_maps = self.dense_block(
                            inputs=inputs,
                            index=index,
                            training=training
                        )

                        return feature_maps

                    elif index == 1:

                        feature_maps = grow(inputs, index - 1)

                        images = self.color_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        feature_maps = self.deconv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        return feature_maps, images

                    elif index == self.num_layers - 1:

                        feature_maps, images = grow(inputs, index - 1)

                        old_images = ops.upsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format,
                            dynamic=True
                        )

                        new_images = self.color_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        images = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, ceiled_coloring_index): lambda: old_images,
                                tf.less(index, ceiled_coloring_index): lambda: new_images
                            },
                            default=lambda: lerp(
                                a=old_images,
                                b=new_images,
                                t=coloring_index - (index - 1)
                            ),
                            exclusive=True
                        )

                        return images

                    else:

                        feature_maps, images = grow(inputs, index - 1)

                        old_images = ops.upsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format,
                            dynamic=True
                        )

                        new_images = self.color_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        images = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, ceiled_coloring_index): lambda: old_images,
                                tf.less(index, ceiled_coloring_index): lambda: new_images
                            },
                            default=lambda: lerp(
                                a=old_images,
                                b=new_images,
                                t=coloring_index - (index - 1)
                            ),
                            exclusive=True
                        )

                        feature_maps = self.deconv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        return feature_maps, images

            return grow(inputs, self.num_layers - 1)

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        raise NotImplementedError()

    def deconv2d_block(self, inputs, index, training, name="deconv2d_block", reuse=None):

        raise NotImplementedError()

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        raise NotImplementedError()


class Discriminator(object):

    def __init__(self, min_resolution, max_resolution, min_filters, max_filters, data_format):

        if (max_resolution // min_resolution) != (max_filters // min_filters):
            raise ValueError("Invalid number of filters")

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.data_format = data_format
        self.num_layers = int(np.log2(max_resolution // min_resolution)) + 2

    def __call__(self, inputs, coloring_index, training, name="discriminator", reuse=None):

        floored_coloring_index = tf.cast(tf.floor(coloring_index), tf.int32)

        with tf.variable_scope(name, reuse=reuse):

            #========================================================================#
            # very complicated but efficient architecture
            # each layer has two output paths: feature_maps and images
            # whether which path is evaluated at runtime
            # depends on variable "coloring_index"
            # but, all possible pathes must be constructed at compile time
            #========================================================================#
            def grow(feature_maps, images, index):

                with tf.variable_scope("layer_{}".format(index)):

                    if index == 0:

                        logits = self.dense_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        return logits

                    elif index == 1:

                        old_feature_maps = self.color_block(
                            inputs=images,
                            index=index,
                            training=training
                        )

                        new_feature_maps = self.conv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        feature_maps = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, floored_coloring_index): lambda: old_feature_maps,
                                tf.less(index, floored_coloring_index): lambda: new_feature_maps
                            },
                            default=lambda: lerp(
                                a=old_feature_maps,
                                b=new_feature_maps,
                                t=coloring_index - index
                            ),
                            exclusive=True
                        )

                        return grow(feature_maps, None, index - 1)

                    elif index == self.num_layers - 1:

                        feature_maps = self.color_block(
                            inputs=images,
                            index=index,
                            training=training
                        )

                        images = ops.downsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format
                        )

                        return grow(feature_maps, images, index - 1)

                    else:

                        old_feature_maps = self.color_block(
                            inputs=images,
                            index=index,
                            training=training
                        )

                        new_feature_maps = self.conv2d_block(
                            inputs=feature_maps,
                            index=index,
                            training=training
                        )

                        feature_maps = tf.case(
                            pred_fn_pairs={
                                tf.greater(index, floored_coloring_index): lambda: old_feature_maps,
                                tf.less(index, floored_coloring_index): lambda: new_feature_maps
                            },
                            default=lambda: lerp(
                                a=old_feature_maps,
                                b=new_feature_maps,
                                t=coloring_index - index
                            ),
                            exclusive=True
                        )

                        images = ops.downsampling2d(
                            inputs=images,
                            factors=[2, 2],
                            data_format=self.data_format
                        )

                        return grow(feature_maps, images, index - 1)

            return grow(None, inputs, self.num_layers - 1)

    def dense_block(self, inputs, index, training, name="dense_block", reuse=None):

        raise NotImplementedError()

    def conv2d_block(self, inputs, index, training, name="conv2d_block", reuse=None):

        raise NotImplementedError()

    def color_block(self, inputs, index, training, name="color_block", reuse=None):

        raise NotImplementedError()
