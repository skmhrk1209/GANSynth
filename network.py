import tensorflow as tf
import numpy as np
from ops import *


def scale(inputs, in_min, in_max, out_min, out_max):
    inputs = out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)
    return inputs


def lerp(a, b, t):
    return t * a + (1 - t) * b


class PGGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels, apply_spectral_norm):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.apply_spectral_norm = apply_spectral_norm

        def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)

        self.min_depth = log2(self.min_resolution // self.min_resolution)
        self.max_depth = log2(self.max_resolution // self.min_resolution)

    def generator(self, latents, labels, training, progress, name="ganerator", reuse=None):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):

            with tf.variable_scope("conv_block_{}x{}".format(*resolution(depth)), reuse=reuse):
                if depth == self.min_depth:
                    inputs = tf.reshape(inputs, [-1, inputs.shape[1], 1, 1])
                    # inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv_upscale"):
                        inputs = conv2d_transpose(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=resolution(depth).tolist(),
                            strides=resolution(depth).tolist()
                        )
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        # inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3]
                        )
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        # inputs = pixel_norm(inputs)
                else:
                    with tf.variable_scope("conv_upscale"):
                        inputs = conv2d_transpose(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3],
                            strides=[2, 2]
                        )
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        # inputs = pixel_norm(inputs)
                    with tf.variable_scope("conv"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth),
                            kernel_size=[3, 3]
                        )
                        inputs = conditional_batch_norm(
                            inputs=inputs,
                            labels=labels,
                            training=training
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                        # inputs = pixel_norm(inputs)
                return inputs

        def color_block(inputs, depth, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("color_block_{}x{}".format(*resolution(depth)), reuse=reuse):
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
            ''' depthに対応する層によって特徴マップを画像として生成
            Args:
                feature_maps: 1つ浅い層から受け取った特徴マップ
                depth: 現在見ている層の深さ（解像度が上がる方向に深くなると定義）
            Returns:
                images: 最終的な生成画像
            '''

            # 現在より深い層によって生成された画像（ここで再帰）
            def high_resolution_images():
                return grow(conv_block(feature_maps, depth), depth + 1)

            # 現在の層の解像度で生成した画像を最終解像度までupscaleする
            def middle_resolution_images():
                return upscale2d(
                    inputs=color_block(conv_block(feature_maps, depth), depth),
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

        with tf.variable_scope(name, reuse=reuse):
            return grow(tf.concat([latents, labels], axis=-1), self.min_depth)

    def discriminator(self, images, labels, training, progress, name="dicriminator", reuse=None):

        def resolution(depth):
            return self.min_resolution << depth

        def channels(depth):
            return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def conv_block(inputs, depth, reuse=tf.AUTO_REUSE):

            with tf.variable_scope("conv_block_{}x{}".format(*resolution(depth)), reuse=reuse):
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
                    with tf.variable_scope("conv_downscale"):
                        inputs = conv2d(
                            inputs=inputs,
                            filters=channels(depth - 1),
                            kernel_size=resolution(depth).tolist(),
                            strides=resolution(depth).tolist(),
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                        inputs = tf.nn.leaky_relu(inputs)
                    inputs = tf.squeeze(inputs, axis=[2, 3])
                    with tf.variable_scope("logits"):
                        logits = dense(
                            inputs=inputs,
                            units=1,
                            apply_spectral_norm=self.apply_spectral_norm
                        )
                    with tf.variable_scope("projection"):
                        labels = embedding(
                            inputs=labels,
                            units=inputs.shape[1],
                            apply_spectral_norm=apply_spectral_norm
                        )
                        labels = tf.reduce_sum(
                            input_tensor=inputs * labels,
                            axis=1,
                            keepdims=True
                        )
                    inputs = logits + labels
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
            with tf.variable_scope("color_block_{}x{}".format(*resolution(depth)), reuse=reuse):
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
            ''' depthに対応する層によって画像を特徴マップとして取り込む
            Args:
                images: 入力画像（depthに関わらず一定）
                depth: 現在見ている層の深さ（解像度が上がる方向に深くなると定義）
            Returns:
                feature_maps: 1つ浅い層に渡す特徴マップ
            '''

            # 現在より深い層によって取り込まれた特徴マップ（ここで再帰）
            def high_resolution_feature_maps():
                return conv_block(grow(images, depth + 1), depth)

            # 現在の層の解像度までdownscaleした後，特徴マップとして取り込む
            def middle_resolution_feature_maps():
                return conv_block(color_block(downscale2d(
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

        with tf.variable_scope(name, reuse=reuse):
            return grow(images, self.min_depth)
