import tensorflow as tf
import numpy as np
import skimage
import metrics
import pathlib


class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images, labels = real_input_fn()
        fake_latents = fake_input_fn()
        # =========================================================================================
        fake_images = generator(fake_latents, labels)
        # =========================================================================================
        real_features, real_logits = discriminator(real_images, labels)
        fake_features, fake_logits = discriminator(fake_images, labels)
        # =========================================================================================
        # Non-Saturating Loss + Mode-Seeking Loss + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis]
        # (https://arxiv.org/pdf/1903.05628.pdf)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits)
        # gradient-based mode-seeking loss
        if hyper_params.mode_seeking_loss_weight:
            latent_gradients = tf.gradients(fake_images, [fake_latents])[0]
            mode_seeking_losses = 1 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1e-6)
            generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += real_gradient_penalties * hyper_params.real_gradient_penalty_weight
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            fake_gradients = tf.gradients(fake_logits, [fake_images])[0]
            fake_gradient_penalties = tf.reduce_sum(tf.square(fake_gradients), axis=[1, 2, 3])
            discriminator_losses += fake_gradient_penalties * hyper_params.fake_gradient_penalty_weight
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
        self.real_magnitude_spectrograms = real_images[:, 0, ..., tf.newaxis]
        self.fake_magnitude_spectrograms = fake_images[:, 0, ..., tf.newaxis]
        self.real_instantaneous_frequencies = real_images[:, 1, ..., tf.newaxis]
        self.fake_instantaneous_frequencies = fake_images[:, 1, ..., tf.newaxis]
        self.real_features = real_features
        self.fake_features = fake_features
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_train_op = generator_train_op
        self.discriminator_train_op = discriminator_train_op

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
                    summary_op=tf.summary.merge(list(map(
                        lambda name_tensor: tf.summary.image(*name_tensor), dict(
                            real_images=self.real_images,
                            fake_images=self.fake_images
                        ).items()
                    )))
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge(list(map(
                        lambda name_tensor: tf.summary.scalar(*name_tensor), dict(
                            generator_loss=self.generator_loss,
                            discriminator_loss=self.discriminator_loss
                        ).items()
                    )))
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        generator_loss=self.generator_loss,
                        discriminator_loss=self.discriminator_loss
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                session.run(self.discriminator_train_op)
                session.run(self.generator_train_op)

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
                            self.tensors.fake_features
                        ])
                    except tf.errors.OutOfRangeError:
                        break

            real_magnitude_spectrograms, fake_magnitude_spectrograms, \
                real_features, fake_features = map(np.concatenate, zip(*generator()))

            def spatial_flatten(inputs):
                return np.reshape(inputs, [-1, np.prod(inputs.shape[1:])])

            tf.logging.info(
                "num_different_bins: {}, frechet_inception_distance: {}".format(
                    metrics.num_different_bins(
                        real_features=spatial_flatten(real_magnitude_spectrograms),
                        fake_features=spatial_flatten(fake_magnitude_spectrograms),
                        num_bins=50,
                        significance_level=0.05
                    ),
                    metrics.frechet_inception_distance(
                        real_features=real_features,
                        fake_features=fake_features
                    )
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

            magnitude_spectrogram_dir = pathlib.Path("samples/magnitude_spectrograms")
            instantaneous_frequency_dir = pathlib.Path("samples/instantaneous_frequencies")

            if not magnitude_spectrogram_dir.exists():
                magnitude_spectrogram_dir.mkdir(parents=True, exist_ok=True)
            if not instantaneous_frequency_dir.exists():
                instantaneous_frequency_dir.mkdir(parents=True, exist_ok=True)

            def generator():
                yield session.run([
                    self.tensors.fake_magnitude_spectrograms,
                    self.tensors.fake_instantaneous_frequencies
                ])

            def unnormalize(inputs, mean, std):
                return inputs * std + mean

            for fake_magnitude_spectrogram, fake_instantaneous_frequency in zip(*map(np.concatenate, zip(*generator()))):
                skimage.io.imsave(
                    fname=magnitude_spectrogram_dir / "{}.jpg".format(len(list(magnitude_spectrogram_dir.glob("*.jpg")))),
                    arr=unnormalize(fake_magnitude_spectrogram, 0.5, 0.5).squeeze()
                )
                skimage.io.imsave(
                    fname=instantaneous_frequency_dir / "{}.jpg".format(len(list(instantaneous_frequency_dir.glob("*.jpg")))),
                    arr=unnormalize(fake_instantaneous_frequency, 0.5, 0.5).squeeze()
                )
