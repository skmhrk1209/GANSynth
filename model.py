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
        fake_latents1, fake_labels = fake_input_fn()
        fake_latents2, fake_labels = fake_input_fn()
        fake_images1 = generator(fake_latents1, fake_labels)
        fake_images2 = generator(fake_latents2, fake_labels)
        # =========================================================================================
        real_features, real_logits = discriminator(real_images, real_labels)
        fake_features1, fake_logits1 = discriminator(fake_images1, fake_labels)
        fake_features2, fake_logits2 = discriminator(fake_images2, fake_labels)
        # =========================================================================================
        # Non-Saturating + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits1)
        generator_losses += tf.nn.softplus(-fake_logits2)
        # mode seeking loss
        if hyper_params.mode_seeking_loss_weight:
            latent_distances = tf.reduce_sum(tf.abs(fake_latents1 - fake_latents2), axis=[1])
            image_distances = tf.reduce_sum(tf.abs(fake_images1 - fake_images2), axis=[1, 2, 3])
            mode_seeking_losses = latent_distances / (image_distances + 1e-6)
            generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits1)
        discriminator_losses += tf.nn.softplus(fake_logits2)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += real_gradient_penalties * hyper_params.real_gradient_penalty_weight
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            fake_gradients1 = tf.gradients(fake_logits1, [fake_images1])[0]
            fake_gradients2 = tf.gradients(fake_logits2, [fake_images2])[0]
            fake_gradient_penalties1 = tf.reduce_sum(tf.square(fake_gradients1), axis=[1, 2, 3])
            fake_gradient_penalties2 = tf.reduce_sum(tf.square(fake_gradients2), axis=[1, 2, 3])
            discriminator_losses += fake_gradient_penalties1 * hyper_params.fake_gradient_penalty_weight
            discriminator_losses += fake_gradient_penalties2 * hyper_params.fake_gradient_penalty_weight
        # -----------------------------------------------------------------------------------------
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
        # tensors and operations used later
        self.operations = Struct(
            discriminator_train_op=discriminator_train_op,
            generator_train_op=generator_train_op
        )
        self.tensors = Struct(
            global_step=tf.train.get_global_step(),
            real_magnitude_spectrograms=real_images[:, 0, ..., tf.newaxis],
            fake_magnitude_spectrograms=fake_images1[:, 0, ..., tf.newaxis],
            real_instantaneous_frequencies=real_images[:, 1, ..., tf.newaxis],
            fake_instantaneous_frequencies=fake_images1[:, 1, ..., tf.newaxis],
            real_features=real_features,
            fake_features=fake_features1,
            real_logits=real_logits,
            fake_logits=fake_logits1,
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
                "num_different_bins: {}, frechet_inception_distance: {}, inception_score: {}".format(
                    metrics.num_different_bins(
                        real_features=np.reshape(real_magnitude_spectrograms, [-1, np.prod(real_magnitude_spectrograms.shape[1:])]),
                        fake_features=np.reshape(fake_magnitude_spectrograms, [-1, np.prod(fake_magnitude_spectrograms.shape[1:])]),
                        num_bins=50,
                        significance_level=0.05
                    ),
                    metrics.frechet_inception_distance(real_features, fake_features),
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

            for fake_magnitude_spectrogram, fake_instantaneous_frequency in zip(*session.run([self.tensors.fake_magnitude_spectrograms, self.tensors.fake_instantaneous_frequencies])):
                skimage.io.imsave(
                    fname=magnitude_spectrogram_dir / "{}.jpg".format(len(list(magnitude_spectrogram_dir.glob("*.jpg")))),
                    arr=linear_map(fake_magnitude_spectrogram, -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255).squeeze()
                )
                skimage.io.imsave(
                    fname=instantaneous_frequency_dir / "{}.jpg".format(len(list(instantaneous_frequency_dir.glob("*.jpg")))),
                    arr=linear_map(fake_instantaneous_frequency, -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255).squeeze()
                )
