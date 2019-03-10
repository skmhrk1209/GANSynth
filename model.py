import tensorflow as tf
import numpy as np
import skimage
import os
from utils import Struct


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):

        # =========================================================================================
        real_images, real_labels = real_input_fn()
        fake_latents, fake_labels = fake_input_fn()
        # =========================================================================================
        fake_images = generator(fake_latents, fake_labels)
        # =========================================================================================
        real_logits = discriminator(real_images, real_labels)
        fake_logits = discriminator(fake_images, fake_labels)
        # =========================================================================================
        # Non-Saturating + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------
        # generator
        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits)
        # -----------------------------------------------------------------------------------------
        # discriminator
        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_zero_centered_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += 0.5 * hyper_params.real_zero_centered_gradient_penalty_weight * real_gradient_penalties
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_zero_centered_gradient_penalty_weight:
            fake_gradients = tf.gradients(fake_logits, [fake_images])[0]
            fake_gradient_penalties = tf.reduce_sum(tf.square(fake_gradients), axis=[1, 2, 3])
            discriminator_losses += 0.5 * hyper_params.fake_zero_centered_gradient_penalty_weight * fake_gradient_penalties
        '''
        # =========================================================================================
        # WGAN-GP + ACGAN
        # [Improved Training of Wasserstein GANs]
        # (https://arxiv.org/pdf/1704.00028.pdf)
        # [Conditional Image Synthesis With Auxiliary Classifier GANs]
        # (https://arxiv.org/pdf/1610.09585.pdf)
        # -----------------------------------------------------------------------------------------
        # generator
        # wasserstein loss
        generator_losses = -fake_logits[:, 0]
        # auxiliary classification loss
        if hyper_params.generator_auxiliary_classification_weight:
            generator_auxiliary_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=fake_labels, logits=fake_logits[:, 1:])
            generator_losses += hyper_params.generator_auxiliary_classification_weight * generator_auxiliary_classification_losses
        # -----------------------------------------------------------------------------------------
        # discriminator
        # wasserstein loss
        discriminator_losses = -real_logits[:, 0] + fake_logits[:, 0]
        # one-centered gradient penalty
        if hyper_params.one_centered_gradient_penalty_weight:
            def lerp(a, b, t): return t * a + (1. - t) * b
            coefficients = tf.random_uniform([tf.shape(real_images)[0], 1, 1, 1])
            interpolated_images = lerp(real_images, fake_images, coefficients)
            interpolated_logits = discriminator(interpolated_images, real_labels)
            interpolated_gradients = tf.gradients(interpolated_logits[:, 0], [interpolated_images])[0]
            interpolated_gradient_penalties = tf.square(1. - tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]) + 1e-8))
            discriminator_losses += hyper_params.one_centered_gradient_penalty_weight * interpolated_gradient_penalties
        # auxiliary classification loss
        if hyper_params.discriminator_auxiliary_classification_weight:
            discriminator_auxiliary_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=real_labels, logits=real_logits[:, 1:])
            discriminator_auxiliary_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(labels=fake_labels, logits=fake_logits[:, 1:])
            discriminator_losses += hyper_params.discriminator_auxiliary_classification_weight * discriminator_auxiliary_classification_losses
        '''
        # =========================================================================================
        # losss reduction
        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)
        # =========================================================================================
        generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )
        # -----------------------------------------------------------------------------------------
        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        # =========================================================================================
        with tf.variable_scope("", reuse=True):
            global_step = tf.get_variable(name="global_step", dtype=tf.int32)

        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            # global_step=tf.train.get_or_create_global_step()
            global_step=global_step
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # tensors and operations used later
        self.operations = Struct(
            generator_train_op=generator_train_op,
            discriminator_train_op=discriminator_train_op
        )
        self.tensors = Struct(
            real_images=real_images,
            real_labels=real_labels,
            fake_latents=fake_latents,
            fake_labels=fake_labels,
            fake_images=fake_images,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            global_step=global_step
        )
        # =========================================================================================
        # scaffold
        self.scaffold = tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.tables_initializer(),
            saver=tf.train.Saver(
                max_to_keep=10,
                keep_checkpoint_every_n_hours=12,
            ),
            summary_op=tf.summary.merge([
                tf.summary.image(
                    name="real_log_mel_magnitude_spectrograms",
                    tensor=real_images[:, 0, ..., tf.newaxis],
                    max_outputs=4
                ),
                tf.summary.image(
                    name="real_mel_instantaneous_frequencies",
                    tensor=real_images[:, 1, ..., tf.newaxis],
                    max_outputs=4
                ),
                tf.summary.image(
                    name="fake_log_mel_magnitude_spectrograms",
                    tensor=fake_images[:, 0, ..., tf.newaxis],
                    max_outputs=4
                ),
                tf.summary.image(
                    name="fake_mel_instantaneous_frequencies",
                    tensor=fake_images[:, 1, ..., tf.newaxis],
                    max_outputs=4
                ),
                tf.summary.scalar(
                    name="generator_loss",
                    tensor=generator_loss
                ),
                tf.summary.scalar(
                    name="discriminator_loss",
                    tensor=discriminator_loss
                ),
            ])
        )

    def train(self, model_dir, total_steps, save_checkpoint_steps,
              save_summary_steps, log_step_count_steps, config):

        with tf.train.SingularMonitoredSession(
            scaffold=self.scaffold,
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    scaffold=self.scaffold,
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    scaffold=self.scaffold
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        # global_step=tf.train.get_global_step(),
                        global_step=self.tensors.global_step,
                        generator_loss=self.tensors.generator_loss,
                        discriminator_loss=self.tensors.discriminator_loss
                    ),
                    every_n_iter=log_step_count_steps,
                ),
                tf.train.StepCounterHook(
                    output_dir=model_dir,
                    every_n_steps=log_step_count_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                session.run(self.operations.discriminator_train_op)
                session.run(self.operations.generator_train_op)

    def generate(self, model_dir, sample_dir1, sample_dir2, steps, config):

        if not os.path.exists(sample_dir1):
            os.makedirs(sample_dir1)
        if not os.path.exists(sample_dir2):
            os.makedirs(sample_dir2)

        with tf.train.SingularMonitoredSession(
            scaffold=self.scaffold,
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            i = 0
            for i in range(steps):
                for image in session.run(self.tensors.fake_images):
                    image1 = linear_map(image[0], -1.0, 1.0, 0.0, 1.0)
                    image2 = linear_map(image[1], -1.0, 1.0, 0.0, 1.0)
                    skimage.io.imsave(os.path.join(sample_dir1, "{}.jpg".format(i)), image1.clip(0.0, 1.0))
                    skimage.io.imsave(os.path.join(sample_dir2, "{}.jpg".format(i)), image2.clip(0.0, 1.0))
                    i += 1
