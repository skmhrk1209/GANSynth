import tensorflow as tf
import numpy as np
from ops import *


def log(x, base):
    return tf.log(x) / tf.log(base)


def lerp(a, b, t):
    return t * a + (1.0 - t) * b


class PGGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels, growing_level):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.growing_level = growing_level

        def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)

        self.min_depth = log2(self.min_resolution // self.min_resolution)
        self.max_depth = log2(self.max_resolution // self.min_resolution)

        self.growing_depth = log(1 + ((1 << (self.max_depth + 1)) - 1) * self.growing_level, 2.0)

    def generator(self, latents, labels, name="generator", reuse=tf.AUTO_REUSE):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(f"conv_block_{'x'.join(map(str, resolution(depth)))}", reuse=reuse):
                if depth == self.min_depth:
                    inputs = pixel_norm(inputs)
                    with tf.variable_scope("dense"):
                        inputs = dense(
                            inputs=inputs,
                            units=channels(depth) * resolution(depth).prod(),
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.reshape(
                            tensor=inputs,
                            shape=[-1, channels(depth), *resolution(depth)]
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    return inputs
                else:
                    with tf.variable_scope("upscale_conv"):
                        inputs = conv2d_transpose(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            strides=[2, 2],
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        inputs = pixel_norm(inputs)
                    return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(f"color_block_{'x'.join(map(str, resolution(depth)))}", reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=2,
                        kernel_size=[1, 1],
                        use_bias=True,
                        variance_scale=1.0,
                        scale_weight=True
                    )
                    inputs = tf.nn.tanh(inputs)
                return inputs

        def grow(feature_maps, depth):

            def high_resolution_images():
                return grow(conv_block(feature_maps, depth), depth + 1)

            def middle_resolution_images():
                return upscale2d(
                    inputs=color_block(conv_block(feature_maps, depth), depth),
                    factors=resolution(self.max_depth) // resolution(depth)
                )

            def low_resolution_images():
                return upscale2d(
                    inputs=color_block(feature_maps, depth - 1),
                    factors=resolution(self.max_depth) // resolution(depth - 1)
                )

            if depth == self.min_depth:
                images = tf.cond(
                    pred=tf.greater(self.growing_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=middle_resolution_images
                )
            elif depth == self.max_depth:
                images = tf.cond(
                    pred=tf.greater(self.growing_depth, depth),
                    true_fn=middle_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - self.growing_depth
                    )
                )
            else:
                images = tf.cond(
                    pred=tf.greater(self.growing_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=lambda: lerp(
                        a=low_resolution_images(),
                        b=middle_resolution_images(),
                        t=depth - self.growing_depth
                    )
                )
            return images

        with tf.variable_scope(name, reuse=reuse):
            labels = embedding(
                inputs=labels,
                units=latents.shape[1],
                variance_scale=1.0,
                scale_weight=True
            )
            return grow(tf.concat([latents, labels], axis=1), self.min_depth)

    def discriminator(self, images, labels, name="discriminator", reuse=tf.AUTO_REUSE):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(f"conv_block_{'x'.join(map(str, resolution(depth)))}", reuse=reuse):
                if depth == self.min_depth:
                    inputs = tf.concat([inputs, batch_stddev(inputs)], axis=1)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("dense"):
                        inputs = tf.layers.flatten(inputs)
                        inputs = dense(
                            inputs=inputs,
                            units=channels(depth - 1),
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("logits"):
                        # label conditioning from
                        # [Which Training Methods for GANs do actually Converge?]
                        # (https://arxiv.org/pdf/1801.04406.pdf)
                        inputs = dense(
                            inputs=inputs,
                            units=labels.shape[1],
                            use_bias=True,
                            variance_scale=1.0,
                            scale_weight=True
                        )
                        inputs = tf.gather_nd(
                            params=inputs,
                            indices=tf.where(labels)
                        )
                    return inputs
                else:
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    with tf.variable_scope("conv_downscale"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth - 1),
                            kernel_size=[3, 3],
                            strides=[2, 2],
                            use_bias=True,
                            variance_scale=2.0,
                            scale_weight=True
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(f"color_block_{'x'.join(map(str, resolution(depth)))}", reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=channels(depth),
                        kernel_size=[1, 1],
                        use_bias=True,
                        variance_scale=2.0,
                        scale_weight=True
                    )
                    inputs = tf.nn.leaky_relu(inputs)
                return inputs

        def grow(images, depth):

            def high_resolution_feature_maps():
                return conv_block(grow(images, depth + 1), depth)

            def middle_resolution_feature_maps():
                return conv_block(color_block(downscale2d(
                    inputs=images,
                    factors=resolution(self.max_depth) // resolution(depth)
                ), depth), depth)

            def low_resolution_feature_maps():
                return color_block(downscale2d(
                    inputs=images,
                    factors=resolution(self.max_depth) // resolution(depth - 1)
                ), depth - 1)

            if depth == self.min_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(self.growing_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=middle_resolution_feature_maps
                )
            elif depth == self.max_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(self.growing_depth, depth),
                    true_fn=middle_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=depth - self.growing_depth
                    )
                )
            else:
                feature_maps = tf.cond(
                    pred=tf.greater(self.growing_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=lambda: lerp(
                        a=low_resolution_feature_maps(),
                        b=middle_resolution_feature_maps(),
                        t=depth - self.growing_depth
                    )
                )
            return feature_maps

        with tf.variable_scope(name, reuse=reuse):
            return grow(images, self.min_depth)


class ResNet(object):

    def __init__(self, conv_param, pool_param, residual_params, groups, classes):

        self.conv_param = conv_param
        self.pool_param = pool_param
        self.residual_params = residual_params
        self.groups = groups
        self.classes = classes

    def __call__(self, inputs, name="resnet", reuse=tf.AUTO_REUSE):

        with tf.variable_scope(name, reuse=reuse):

            if self.conv_param:
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=self.conv_param.filters,
                        kernel_size=self.conv_param.kernel_size,
                        strides=self.conv_param.strides,
                        use_bias=True,
                        variance_scale=2.0,
                        apply_weight_standardization=True
                    )

            if self.pool_param:
                inputs = max_pooling2d(
                    inputs=inputs,
                    kernel_size=self.pool_param.kernel_size,
                    strides=self.pool_param.strides
                )

            for i, residual_param in enumerate(self.residual_params):

                for j in range(residual_param.blocks)[:1]:
                    with tf.variable_scope(f"residual_block_{i}_{j}"):
                        inputs = self.residual_block(
                            inputs=inputs,
                            filters=residual_param.filters,
                            strides=residual_param.strides,
                            projection_shortcut=True,
                            groups=self.groups
                        )

                for j in range(residual_param.blocks)[1:]:
                    with tf.variable_scope(f"residual_block_{i}_{j}"):
                        inputs = self.residual_block(
                            inputs=inputs,
                            filters=residual_param.filters,
                            strides=[1, 1],
                            projection_shortcut=False,
                            groups=self.groups
                        )

            with tf.variable_scope("group_normalization"):
                inputs = group_normalization(inputs, groups=self.groups)

            inputs = tf.nn.relu(inputs)

            features = tf.reduce_mean(inputs, axis=[2, 3])

            with tf.variable_scope("logits"):
                logits = dense(
                    inputs=features,
                    units=self.classes,
                    use_bias=True,
                    variance_scale=1.0,
                    apply_weight_standardization=False
                )

            return features, logits

    def residual_block(self, inputs, filters, strides, projection_shortcut, groups):
        """ A single block for ResNet v2, without a bottleneck.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        (https://arxiv.org/pdf/1603.05027.pdf)
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        """

        shortcut = inputs

        with tf.variable_scope("group_normalization_1st"):
            inputs = group_normalization(inputs, groups=groups)

        inputs = tf.nn.relu(inputs)

        if projection_shortcut:
            with tf.variable_scope("projection_shortcut"):
                shortcut = conv2d(
                    inputs=inputs,
                    filters=filters,
                    kernel_size=[1, 1],
                    strides=strides,
                    use_bias=False,
                    variance_scale=2.0,
                    apply_weight_standardization=True
                )

        with tf.variable_scope("conv_1st"):
            inputs = conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=strides,
                use_bias=True,
                variance_scale=2.0,
                apply_weight_standardization=True
            )

        with tf.variable_scope("group_normalization_2nd"):
            inputs = group_normalization(inputs, groups=groups)

        inputs = tf.nn.relu(inputs)

        with tf.variable_scope("conv_2nd"):
            inputs = conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                use_bias=True,
                variance_scale=2.0,
                apply_weight_standardization=True
            )

        inputs += shortcut

        return inputs
