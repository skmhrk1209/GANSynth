import tensorflow as tf
import numpy as np
import skimage
import metrics
from pathlib import Path
from utils import Struct


def lerp(a, b, t):
    return t * a + (1.0 - t) * b


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images, real_labels = real_input_fn()
        fake_latents, fake_labels = fake_input_fn()
        fake_images = generator(fake_latents, fake_labels)
        # =========================================================================================
        real_features, real_adversarial_logits, real_classification_logits = discriminator(real_images, real_labels)
        fake_features, fake_adversarial_logits, fake_classification_logits = discriminator(fake_images, fake_labels)
        real_adversarial_logits = tf.squeeze(real_adversarial_logits, axis=1)
        fake_adversarial_logits = tf.squeeze(fake_adversarial_logits, axis=1)
        # =========================================================================================
        # WGAN-GP + ACGAN
        # [Improved Training of Wasserstein GANs]
        # (https://arxiv.org/pdf/1704.00028.pdf)
        # [Conditional Image Synthesis With Auxiliary Classifier GANs]
        # (https://arxiv.org/pdf/1610.09585.pdf)
        # -----------------------------------------------------------------------------------------
        # generator wasserstein loss
        generator_adversarial_losses = -fake_adversarial_logits
        # generator classification loss
        generator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=fake_labels,
            logits=fake_classification_logits
        )
        generator_losses = \
            generator_adversarial_losses + \
            generator_classification_losses * hyper_params.generator_classification_weight
        # -----------------------------------------------------------------------------------------
        # discriminator wasserstein loss
        discriminator_adversarial_losses = -real_adversarial_logits
        discriminator_adversarial_losses += fake_adversarial_logits
        # discriminator one-centered gradient penalty
        coefficients = tf.random_uniform([tf.shape(real_images)[0], 1, 1, 1])
        interpolated_images = lerp(real_images, fake_images, coefficients)
        _, interpolated_adversarial_logits, _ = discriminator(interpolated_images, real_labels)
        interpolated_gradients = tf.gradients(interpolated_adversarial_logits, [interpolated_images])[0]
        interpolated_gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]) + 1e-8)
        interpolated_gradient_penalties = tf.square(1.0 - interpolated_gradient_norms)
        # discriminator classification loss
        discriminator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=real_labels,
            logits=real_classification_logits
        )
        discriminator_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=fake_labels,
            logits=fake_classification_logits
        )
        discriminator_losses = \
            discriminator_adversarial_losses + \
            interpolated_gradient_penalties * hyper_params.gradient_penalty_weight + \
            discriminator_classification_losses * hyper_params.discriminator_classification_weight
        # -----------------------------------------------------------------------------------------
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
        # tensors and operations used later
        self.operations = Struct(
            discriminator_train_op=discriminator_train_op,
            generator_train_op=generator_train_op
        )
        self.tensors = Struct(
            global_step=tf.train.get_global_step(),
            real_magnitude_spectrograms=real_images[:, 0, ..., tf.newaxis],
            fake_magnitude_spectrograms=fake_images[:, 0, ..., tf.newaxis],
            real_instantaneous_frequencies=real_images[:, 1, ..., tf.newaxis],
            fake_instantaneous_frequencies=fake_images[:, 1, ..., tf.newaxis],
            real_features=real_features,
            fake_features=fake_features,
            real_adversarial_logits=real_adversarial_logits,
            fake_adversarial_logits=fake_adversarial_logits,
            real_classification_logits=real_classification_logits,
            fake_classification_logits=fake_classification_logits,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss
        )

    def train(self, model_dir, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    )
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.rank == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items()
                        if tensor.shape.rank == 0 or tensor.shape.rank == 4
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if tensor.shape.rank == 0
                    },
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                for name, operation in self.operations.items():
                    session.run(operation)

    def evaluate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            def generator():
                while True:
                    try:
                        yield session.run([
                            self.tensors.real_magnitude_spectrograms,
                            self.tensors.fake_magnitude_spectrograms,
                            self.tensors.real_features,
                            self.tensors.fake_features,
                            self.tensors.real_classification_logits,
                            self.tensors.fake_classification_logits
                        ])
                    except tf.errors.OutOfRangeError:
                        break

            real_magnitude_spectrograms, fake_magnitude_spectrograms, real_features, fake_features, \
                real_classification_logits, fake_classification_logits = map(np.concatenate, zip(*generator()))

            tf.logging.info(
                "num_different_bins: {}, frechet_inception_distance: {}, "
                "real_inception_score: {}, fake_inception_score: {}".format(
                    metrics.num_different_bins(np.ravel(real_magnitude_spectrograms), np.ravel(fake_magnitude_spectrograms), num_bins=50),
                    metrics.frechet_inception_distance(real_features, fake_features),
                    metrics.inception_score(real_classification_logits),
                    metrics.inception_score(fake_classification_logits)
                )
            )

    def generate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            magnitude_spectrogram_dir = Path("samples/magnitude_spectrograms")
            instantaneous_frequency_dir = Path("samples/instantaneous_frequencies")

            if not magnitude_spectrogram_dir.exists():
                magnitude_spectrogram_dir.mkdir(parents=True, exist_ok=True)
            if not instantaneous_frequency_dir.exists():
                instantaneous_frequency_dir.mkdir(parents=True, exist_ok=True)

            for fake_magnitude_spectrogram, fake_instantaneous_frequency in session.run([self.tensors.fake_magnitude_spectrograms, self.tensors.fake_instantaneous_frequencies]):
                skimage.io.imsave(
                    fname=magnitude_spectrogram_dir / "{}.jpg".format(len(list(magnitude_spectrogram_dir.glob("*.jpg")))),
                    arr=linear_map(fake_magnitude_spectrogram, -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
                )
                skimage.io.imsave(
                    fname=instantaneous_frequency_dir / "{}.jpg".format(len(list(instantaneous_frequency_dir.glob("*.jpg")))),
                    arr=linear_map(fake_instantaneous_frequency, -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
                )
