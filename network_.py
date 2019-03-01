import tensorflow as tf
import numpy as np
from ops import *


class PGGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels,
                 scale_weight, apply_spectral_norm):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.scale_weight = scale_weight
        self.apply_spectral_norm = apply_spectral_norm

        def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)

        self.min_depth = log2(self.min_resolution // self.min_resolution)
        self.max_depth = log2(self.max_resolution // self.min_resolution)

    def generator(self, latents, labels, training, progress, name="ganerator", reuse=None):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def residual_block(inputs, depth, reuse=tf.AUTO_REUSE):
            ''' A single block for ResNet v2, without a bottleneck.
                Batch normalization then ReLu then convolution as described by:
                Identity Mappings in Deep Residual Networks
                (https://arxiv.org/pdf/1603.05027.pdf)
                by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
            '''
            with tf.variable_scope("residual_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    with tf.variable_scope("dense"):
                        inputs = dense(
                            inputs=inputs,
                            units=channels(depth) * resolution(depth).prod(),
                            use_bias=False
                        )
                        inputs = tf.reshape(
                            tensor=inputs,
                            shape=[-1, channels(depth), *resolution(depth)]
                        )
                    with tf.variable_scope("conditional_batch_norm"):
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                    inputs = tf.nn.relu(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=False,
                            variance_scale=2,
                            scale_weight=self.scale_weight
                        )
                else:
                    with tf.variable_scope("conditional_batch_norm_1st"):
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                    inputs = tf.nn.relu(inputs)
                    # projection shortcut should come after batch norm and relu
                    # since it performs a 1x1 convolution
                    with tf.variable_scope("projection_shortcut"):
                        shortcut = upscale2d(inputs)
                        shortcut = conv2d(
                            inputs=shortcut,
                            filters=channels(depth),
                            kernel_size=[1, 1],
                            use_bias=False,
                            variance_scale=2,
                            scale_weight=self.scale_weight
                        )
                    with tf.variable_scope("conv_1st"):
                        inputs = upscale2d(inputs)
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=False,
                            variance_scale=2,
                            scale_weight=self.scale_weight
                        )
                    with tf.variable_scope("conditional_batch_norm_2nd"):
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                    inputs = tf.nn.relu(inputs)
                    with tf.variable_scope("conv_2nd"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=False,
                            variance_scale=2,
                            scale_weight=self.scale_weight
                        )
                    inputs += shortcut
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                with tf.variable_scope("batch_norm"):
                    inputs = batch_norm(
                        inputs=inputs,
                        training=training
                    )
                inputs = tf.nn.relu(inputs)
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=2,
                        kernel_size=[1, 1],
                        use_bias=True,
                        variance_scale=1,
                        scale_weight=self.scale_weight
                    )
                inputs = tf.nn.tanh(inputs)
                return inputs

        def lerp(a, b, t): return t * a + (1 - t) * b

        def grow(feature_maps, depth):
            ''' depthに対応する層によって特徴マップを画像として生成
            Args:
                feature_maps: 1つ浅い層から受け取った特徴マップ
                depth: 現在見ている層の深さ（解像度が上がる方向に深くなると定義）
            Returns:
                images: 最終的な生成画像
            '''

            # 現在より深い層によって生成された画像（ここで再帰）
            def high_resolution_images():
                return grow(residual_block(feature_maps, depth), depth + 1)

            # 現在の層の解像度で生成した画像を最終解像度までupscaleする
            def middle_resolution_images():
                return upscale2d(
                    inputs=color_block(residual_block(feature_maps, depth), depth),
                    factors=resolution(self.max_depth) // resolution(depth)
                )

            # 1つ浅い層の解像度で生成した画像を最終解像度までupscaleする
            def low_resolution_images():
                return upscale2d(
                    inputs=color_block(feature_maps, depth - 1),
                    factors=resolution(self.max_depth) // resolution(depth - 1)
                )

            # 最も浅い層はlow_resolution_imagesは選択肢にない
            if depth == self.min_depth:
                images = tf.cond(
                    pred=tf.greater(out_depth, depth),
                    true_fn=high_resolution_images,
                    false_fn=middle_resolution_images
                )
            # 最も深い層はhigh_resolution_imagesは選択肢にない
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
            # それ以外は以下のいずれかを出力する
            # 1. high_resolution_images
            # 2. low_resolution_imagesとmiddle_resolution_imagesの線形補間
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

        out_depth = scale(progress, 0.0, 1.0, self.min_depth, self.max_depth)

        with tf.variable_scope(name, reuse=reuse):
            return grow(tf.concat([latents, labels], axis=-1), self.min_depth)

    def discriminator(self, images, labels, training, progress, name="dicriminator", reuse=None):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def residual_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("residual_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    inputs = tf.nn.relu(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2,
                            scale_weight=self.scale_weight,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                    inputs = tf.nn.relu(inputs)
                    # global pooling
                    inputs = tf.reduce_sum(inputs, axis=[2, 3])
                    with tf.variable_scope("logits"):
                        logits = dense(
                            inputs=inputs,
                            units=1,
                            use_bias=True,
                            variance_scale=1,
                            scale_weight=self.scale_weight,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                    # projection discriminator
                    with tf.variable_scope("projections"):
                        embedded = embedding(
                            inputs=labels,
                            units=inputs.shape[1],
                            variance_scale=1,
                            scale_weight=self.scale_weight,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        projections = tf.reduce_sum(
                            input_tensor=inputs * embedded,
                            axis=1,
                            keepdims=True
                        )
                    inputs = logits + projections
                else:
                    inputs = tf.nn.relu(inputs)
                    # projection shortcut should come after batch norm and relu
                    # since it performs a 1x1 convolution
                    with tf.variable_scope("projection_shortcut"):
                        shortcut = conv2d(
                            inputs=inputs,
                            filters=channels(depth - 1),
                            kernel_size=[1, 1],
                            use_bias=False,
                            variance_scale=2,
                            scale_weight=self.scale_weight,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        shortcut = downscale2d(shortcut)
                    with tf.variable_scope("conv_1st"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2,
                            scale_weight=self.scale_weight,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                    inputs = tf.nn.relu(inputs)
                    with tf.variable_scope("conv_2nd"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth - 1),
                            kernel_size=[3, 3],
                            use_bias=True,
                            variance_scale=2,
                            scale_weight=self.scale_weight,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = downscale2d(inputs)
                    inputs += shortcut
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                with tf.variable_scope("conv"):
                    inputs = conv2d(
                        inputs=inputs,
                        filters=channels(depth),
                        kernel_size=[1, 1],
                        use_bias=True,
                        variance_scale=2,
                        scale_weight=self.scale_weight,
                        apply_spectral_norm=self.apply_spectral_norm
                    )
                return inputs

        def lerp(a, b, t): return t * a + (1 - t) * b

        def grow(images, depth):
            ''' depthに対応する層によって画像を特徴マップとして取り込む
            Args:
                images: 入力画像（depthに関わらず一定）
                depth: 現在見ている層の深さ（解像度が上がる方向に深くなると定義）
            Returns:
                feature_maps: 1つ浅い層に渡す特徴マップ
            '''

            # 現在より深い層によって取り込まれた特徴マップ（ここで再帰）
            def high_resolution_feature_maps():
                return residual_block(grow(images, depth + 1), depth)

            # 現在の層の解像度までdownscaleした後，特徴マップとして取り込む
            def middle_resolution_feature_maps():
                return residual_block(color_block(downscale2d(
                    inputs=images,
                    factors=resolution(self.max_depth) // resolution(depth)
                ), depth), depth)

            # 1つ浅い層の解像度までdownscaleした後，特徴マップとして取り込む
            def low_resolution_feature_maps():
                return color_block(downscale2d(
                    inputs=images,
                    factors=resolution(self.max_depth) // resolution(depth - 1)
                ), depth - 1)

            # 最も浅い層はlow_resolution_feature_mapsは選択肢にない
            if depth == self.min_depth:
                feature_maps = tf.cond(
                    pred=tf.greater(in_depth, depth),
                    true_fn=high_resolution_feature_maps,
                    false_fn=middle_resolution_feature_maps
                )
            # 最も深い層はhigh_resolution_feature_mapsは選択肢にない
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
            # それ以外は以下のいずれかを出力する
            # 1. high_resolution_feature_maps
            # 2. low_resolution_feature_mapsとmiddle_resolution_feature_mapsの線形補間
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

        in_depth = scale(progress, 0.0, 1.0, self.min_depth, self.max_depth)

        with tf.variable_scope(name, reuse=reuse):
            return grow(images, self.min_depth)
