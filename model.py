import tensorflow as tf
import numpy as np
import itertools
import functools
import os


def lerp(a, b, t): return t * a + (1. - t) * b


class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params, model_dir):

        # =========================================================================================
        # input_fn for real data and fake data
        self.real_images, self.real_labels = real_input_fn()
        self.latents, self.fake_labels = fake_input_fn()
        # =========================================================================================
        # generated fake data
        self.fake_images = generator(self.latents, self.fake_labels)
        # =========================================================================================
        # logits for real data and fake data
        self.real_logits = discriminator(self.real_images, self.real_labels)
        self.fake_logits = discriminator(self.fake_images, self.fake_labels)
        # =========================================================================================
        # Non-Saturating + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------
        # generator
        # non-saturating loss
        self.generator_losses = tf.nn.softplus(-self.fake_logits)
        # -----------------------------------------------------------------------------------------
        # discriminator
        # non-saturating loss
        self.discriminator_losses = tf.nn.softplus(-self.real_logits)
        self.discriminator_losses += tf.nn.softplus(self.fake_logits)
        # zero-centerd gradient penalty
        if hyper_params.real_zero_centered_gp_weight:
            gradients = tf.gradients(self.real_logits, [self.real_images])[0]
            real_gradient_penalty = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
            self.discriminator_losses += 0.5 * hyper_params.real_zero_centered_gp_weight * real_gradient_penalty
        # zero-centerd gradient penalty
        if hyper_params.fake_zero_centered_gp_weight:
            gradients = tf.gradients(self.fake_logits, [self.fake_images])[0]
            fake_gradient_penalty = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
            self.discriminator_losses += 0.5 * hyper_params.fake_zero_centered_gp_weight * fake_gradient_penalty
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
        self.generator_losses = -self.fake_logits[:, 0]
        # auxiliary classification loss
        if hyper_params.generator_acgan_weight:
            generator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(self.fake_labels, self.fake_logits[:, 1:])
            self.generator_losses += hyper_params.generator_acgan_weight * generator_classification_losses
        # -----------------------------------------------------------------------------------------
        # discriminator
        # wasserstein loss
        self.discriminator_losses = self.fake_logits[:, 0] - self.real_logits[:, 0]
        # one-centered gradient penalty
        if hyper_params.one_centered_gp_weight:
            coefficients = tf.random_uniform([tf.shape(self.real_images)[0], 1, 1, 1])
            interpolated_images = lerp(self.real_images, self.fake_images, coefficients)
            interpolated_logits = discriminator(interpolated_images, self.real_labels)
            gradients = tf.gradients(interpolated_logits[:, 0], [interpolated_images])[0]
            gradient_penalties = tf.square(tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-8) - 1.)
            self.discriminator_losses += hyper_params.one_centered_gp_weight * gradient_penalties
        # auxiliary classification loss
        if hyper_params.discriminator_acgan_weight:
            discriminator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(self.real_labels, self.real_logits[:, 1:])
            discriminator_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(self.fake_labels, self.fake_logits[:, 1:])
            self.discriminator_losses += hyper_params.discriminator_acgan_weight * discriminator_classification_losses
        '''
        # =========================================================================================
        # training op for generator and discriminator
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
        # -----------------------------------------------------------------------------------------
        with tf.variable_scope("", reuse=True):
            self.global_step = tf.get_variable(name="global_step", dtype=tf.int32)
        # -----------------------------------------------------------------------------------------
        self.discriminator_train_op = discriminator_optimizer.minimize(
            loss=tf.reduce_mean(self.discriminator_losses),
            var_list=discriminator_variables
        )
        self.generator_train_op = generator_optimizer.minimize(
            loss=tf.reduce_mean(self.generator_losses),
            var_list=generator_variables,
            global_step=self.global_step
        )
        # =========================================================================================
        # utilities
        self.model_dir = model_dir
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge([
            tf.summary.image("real_log_mel_magnitude_spectrograms", self.real_images[:, 0, ..., tf.newaxis], max_outputs=4),
            tf.summary.image("real_mel_instantaneous_frequencies", self.real_images[:, 1, ..., tf.newaxis], max_outputs=4),
            tf.summary.image("fake_log_mel_magnitude_spectrograms", self.fake_images[:, 0, ..., tf.newaxis], max_outputs=4),
            tf.summary.image("fake_mel_instantaneous_frequencies", self.fake_images[:, 1, ..., tf.newaxis], max_outputs=4),
            tf.summary.scalar("discriminator_loss", tf.reduce_mean(self.discriminator_losses)),
            tf.summary.scalar("generator_loss", tf.reduce_mean(self.generator_losses))
        ])

    def initialize(self):

        session = tf.get_default_session()
        session.run(tf.tables_initializer())

        checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if checkpoint:
            self.saver.restore(session, checkpoint)
            tf.logging.info("global variables in {} restored".format(self.model_dir))
        else:
            session.run(tf.global_variables_initializer())
            tf.logging.info("global variables in {} initialized".format(self.model_dir))

    def train(self, total_steps):

        session = tf.get_default_session()
        writer = tf.summary.FileWriter(self.model_dir, session.graph)

        while True:

            global_step = session.run(self.global_step)

            session.run(self.discriminator_train_op)
            session.run(self.generator_train_op)

            if global_step % 100 == 0:

                discriminator_loss, generator_loss = session.run(
                    [self.discriminator_loss, self.generator_loss]
                )
                tf.logging.info("global_step: {}, discriminator_loss: {:.2f}, generator_loss: {:.2f}".format(
                    self.global_step, self.discriminator_loss, self.generator_loss
                ))

                writer.add_summary(
                    summary=session.run(self.summary),
                    global_step=global_step
                )

                if global_step % 1000 == 0:

                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join(self.model_dir, "model.ckpt"),
                        global_step=global_step
                    )

            if global_step == total_steps:
                break
