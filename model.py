import tensorflow as tf
import numpy as np
import scipy as sp
import skimage
from pathlib import Path
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
        real_features, real_logits = discriminator(real_images, real_labels)
        fake_features, fake_logits = discriminator(fake_images, fake_labels)
        '''
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
            interpolated_features, interpolated_logits = discriminator(interpolated_images, real_labels)
            interpolated_gradients = tf.gradients(interpolated_logits[:, 0], [interpolated_images])[0]
            interpolated_gradient_penalties = tf.square(1. - tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]) + 1e-8))
            discriminator_losses += hyper_params.one_centered_gradient_penalty_weight * interpolated_gradient_penalties
        # auxiliary classification loss
        if hyper_params.discriminator_auxiliary_classification_weight:
            discriminator_auxiliary_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=real_labels, logits=real_logits[:, 1:])
            discriminator_auxiliary_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(labels=fake_labels, logits=fake_logits[:, 1:])
            discriminator_losses += hyper_params.discriminator_auxiliary_classification_weight * discriminator_auxiliary_classification_losses
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
        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # frechet_classifier_distance
        frechet_classifier_distance = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(real_features, fake_features)
        # =========================================================================================
        # tensors and operations used later
        self.operations = Struct(
            generator_train_op=generator_train_op,
            discriminator_train_op=discriminator_train_op
        )
        self.tensors = Struct(
            real_images=real_images,
            real_labels=real_labels,
            real_features=real_features,
            real_logits=real_logits,
            fake_latents=fake_latents,
            fake_labels=fake_labels,
            fake_images=fake_images,
            fake_features=fake_features,
            fake_logits=fake_logits,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            frechet_classifier_distance=frechet_classifier_distance
        )
        # =========================================================================================
        # scaffold
        self.scaffold = tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.group(
                tf.local_variables_initializer(),
                tf.tables_initializer()
            ),
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
                tf.summary.scalar(
                    name="frechet_classifier_distance",
                    tensor=frechet_classifier_distance
                )
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
                        global_step=tf.train.get_global_step(),
                        generator_loss=self.tensors.generator_loss,
                        discriminator_loss=self.tensors.discriminator_loss,
                        frechet_classifier_distance=self.tensors.frechet_classifier_distance
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

    def evaluate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=self.scaffold,
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            real_features = []
            fake_features = []

            try:
                while True:
                    real_features += list(session.run(self.tensors.real_features))
                    fake_features += list(session.run(self.tensors.fake_features))
            except tf.errors.OutOfRangeError:
                pass

            def frechet_classifier_distance(real_features, fake_features):
                real_features = np.asanyarray(real_features)
                fake_features = np.asanyarray(fake_features)
                real_mean = np.mean(real_features, axis=0)
                fake_mean = np.mean(fake_features, axis=0)
                real_cov = np.cov(real_features, rowvar=False)
                fake_cov = np.cov(fake_features, rowvar=False)
                mean_cov = sp.linalg.sqrtm(np.dot(real_cov, fake_cov))
                return np.sum((real_mean - fake_mean) ** 2) + np.trace(real_cov + fake_cov - 2 * mean_cov)

            tf.logging.info("frechet_classifier_distance: {}".format(frechet_classifier_distance(real_features, fake_features)))

    def generate(self, model_dir, sample_dir1, sample_dir2, config):

        sample_dir1 = Path(sample_dir1)
        sample_dir2 = Path(sample_dir2)

        if not sample_dir1.exists():
            sample_dir1.mkdir()
        if not sample_dir2.exists():
            sample_dir2.mkdir()

        with tf.train.SingularMonitoredSession(
            scaffold=self.scaffold,
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            for image in enumerate(session.run(self.tensors.fake_images)):
                skimage.io.imsave(
                    fname=sample_dir1 / "{}.jpg".format(len(list(sample_dir1.glob("*.jpg")))),
                    arr=linear_map(image[0], -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
                )
                skimage.io.imsave(
                    fname=sample_dir2 / "{}.jpg".format(len(list(sample_dir2.glob("*.jpg")))),
                    arr=linear_map(image[1], -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
                )
