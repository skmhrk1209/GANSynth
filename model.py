import tensorflow as tf
from utils import Struct


def lerp(a, b, t): return t * a + (1.0 - t) * b


class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images, real_labels = real_input_fn()
        fake_latents, fake_labels = fake_input_fn()
        fake_images = generator(fake_latents, fake_labels)
        # =========================================================================================
        real_features, real_adversarial_logits, train_real_classification_logits = discriminator(real_images, real_labels)
        fake_features, fake_adversarial_logits, train_fake_classification_logits = discriminator(fake_images, fake_labels)
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
            logits=train_fake_classification_logits
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
            logits=train_real_classification_logits
        )
        discriminator_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=fake_labels,
            logits=train_fake_classification_logits
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
            real_instantaneous_frequencies=real_images[:, 1, ..., tf.newaxis],
            fake_magnitude_spectrograms=fake_images[:, 0, ..., tf.newaxis],
            fake_instantaneous_frequencies=fake_images[:, 1, ..., tf.newaxis],
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
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items()
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if tensor.shape.ndims == 0
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
