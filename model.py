import tensorflow as tf
import numpy as np
import scipy as sp
import skimage
import metrics
from utils import Struct


def lerp(a, b, t): return t * a + (1.0 - t) * b


class GANSynth(object):

    def __init__(self, generator, discriminator, train_real_input_fn, train_fake_input_fn,
                 valid_real_input_fn, valid_fake_input_fn, hyper_params):
        # =========================================================================================
        train_real_images, train_real_labels = train_real_input_fn()
        train_fake_latents, train_fake_labels = train_fake_input_fn()
        # -----------------------------------------------------------------------------------------
        valid_real_images, valid_real_labels = valid_real_input_fn()
        valid_fake_latents, valid_fake_labels = valid_fake_input_fn()
        # =========================================================================================
        train_fake_images = generator(train_fake_latents, train_fake_labels)
        # -----------------------------------------------------------------------------------------
        valid_fake_images = generator(valid_fake_latents, valid_fake_labels)
        # =========================================================================================
        train_real_features, train_real_adversarial_logits, train_real_classification_logits = discriminator(train_real_images, train_real_labels)
        train_fake_features, train_fake_adversarial_logits, train_fake_classification_logits = discriminator(train_fake_images, train_fake_labels)
        train_real_adversarial_logits = tf.squeeze(train_real_adversarial_logits, axis=1)
        train_fake_adversarial_logits = tf.squeeze(train_fake_adversarial_logits, axis=1)
        # -----------------------------------------------------------------------------------------
        valid_real_features, valid_real_adversarial_logits, valid_real_classification_logits = discriminator(valid_real_images, valid_real_labels)
        valid_fake_features, valid_fake_adversarial_logits, valid_fake_classification_logits = discriminator(valid_fake_images, valid_fake_labels)
        valid_real_adversarial_logits = tf.squeeze(valid_real_adversarial_logits, axis=1)
        valid_fake_adversarial_logits = tf.squeeze(valid_fake_adversarial_logits, axis=1)
        # =========================================================================================
        # WGAN-GP + ACGAN
        # [Improved Training of Wasserstein GANs]
        # (https://arxiv.org/pdf/1704.00028.pdf)
        # [Conditional Image Synthesis With Auxiliary Classifier GANs]
        # (https://arxiv.org/pdf/1610.09585.pdf)
        # -----------------------------------------------------------------------------------------
        # wasserstein loss
        train_generator_adversarial_losses = -train_fake_adversarial_logits
        # auxiliary classification loss
        train_generator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=train_fake_labels,
            logits=train_fake_classification_logits
        )
        train_generator_losses = train_generator_adversarial_losses + \
            hyper_params.generator_classification_weight * train_generator_classification_losses
        # -----------------------------------------------------------------------------------------
        # wasserstein loss
        train_discriminator_adversarial_losses = -train_real_adversarial_logits + train_fake_adversarial_logits
        # one-centered gradient penalty
        coefficients = tf.random_uniform([tf.shape(train_real_images)[0], 1, 1, 1])
        train_interpolated_images = lerp(train_real_images, train_fake_images, coefficients)
        _, train_interpolated_adversarial_logits, _ = discriminator(train_interpolated_images, train_real_labels)
        train_interpolated_gradients = tf.gradients(train_interpolated_adversarial_logits, [train_interpolated_images])[0]
        train_interpolated_gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(train_interpolated_gradients), axis=[1, 2, 3]) + 1e-8)
        train_interpolated_gradient_penalties = tf.square(1.0 - train_interpolated_gradient_norms)
        # auxiliary classification loss
        train_discriminator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=train_real_labels,
            logits=train_real_classification_logits
        )
        train_discriminator_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=train_fake_labels,
            logits=train_fake_classification_logits
        )
        train_discriminator_losses = train_discriminator_adversarial_losses + \
            hyper_params.gradient_penalty_weight * train_interpolated_gradient_penalties + \
            hyper_params.discriminator_classification_weight * train_discriminator_classification_losses
        # -----------------------------------------------------------------------------------------
        # losss reduction
        train_generator_loss = tf.reduce_mean(train_generator_losses)
        train_discriminator_loss = tf.reduce_mean(train_discriminator_losses)
        # -----------------------------------------------------------------------------------------
        # wasserstein loss
        valid_generator_adversarial_losses = -valid_fake_adversarial_logits
        # auxiliary classification loss
        valid_generator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=valid_fake_labels,
            logits=valid_fake_classification_logits
        )
        valid_generator_losses = valid_generator_adversarial_losses + \
            hyper_params.generator_classification_weight * valid_generator_classification_losses
        # -----------------------------------------------------------------------------------------
        # wasserstein loss
        valid_discriminator_adversarial_losses = -valid_real_adversarial_logits + valid_fake_adversarial_logits
        # one-centered gradient penalty
        coefficients = tf.random_uniform([tf.shape(valid_real_images)[0], 1, 1, 1])
        valid_interpolated_images = lerp(valid_real_images, valid_fake_images, coefficients)
        _, valid_interpolated_adversarial_logits, _ = discriminator(valid_interpolated_images, valid_real_labels)
        valid_interpolated_gradients = tf.gradients(valid_interpolated_adversarial_logits, [valid_interpolated_images])[0]
        valid_interpolated_gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(valid_interpolated_gradients), axis=[1, 2, 3]) + 1e-8)
        valid_interpolated_gradient_penalties = tf.square(1.0 - valid_interpolated_gradient_norms)
        # auxiliary classification loss
        valid_discriminator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=valid_real_labels,
            logits=valid_real_classification_logits
        )
        valid_discriminator_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=valid_fake_labels,
            logits=valid_fake_classification_logits
        )
        valid_discriminator_losses = valid_discriminator_adversarial_losses + \
            hyper_params.gradient_penalty_weight * valid_interpolated_gradient_penalties + \
            hyper_params.discriminator_classification_weight * valid_discriminator_classification_losses
        # -----------------------------------------------------------------------------------------
        # losss reduction
        valid_generator_loss = tf.reduce_mean(valid_generator_losses)
        valid_discriminator_loss = tf.reduce_mean(valid_discriminator_losses)
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
            loss=train_generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=train_discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # frechet_inception_distance
        train_real_inception_score = tf.contrib.gan.eval.classifier_score_from_logits(train_real_classification_logits)
        train_fake_inception_score = tf.contrib.gan.eval.classifier_score_from_logits(train_fake_classification_logits)
        train_frechet_inception_distance = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(train_real_features, train_fake_features)
        # -----------------------------------------------------------------------------------------
        # frechet_inception_distance
        valid_real_inception_score = tf.contrib.gan.eval.classifier_score_from_logits(valid_real_classification_logits)
        valid_fake_inception_score = tf.contrib.gan.eval.classifier_score_from_logits(valid_fake_classification_logits)
        valid_frechet_inception_distance = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(valid_real_features, valid_fake_features)
        # =========================================================================================
        # tensors and operations used later
        self.operations = Struct(
            discriminator_train_op=discriminator_train_op,
            generator_train_op=generator_train_op
        )
        self.tensors = Struct(
            train_real_magnitude_spectrograms=train_real_images[:, 0, ..., tf.newaxis],
            train_real_instantaneous_frequencies=train_real_images[:, 1, ..., tf.newaxis],
            train_fake_magnitude_spectrograms=train_fake_images[:, 0, ..., tf.newaxis],
            train_fake_instantaneous_frequencies=train_fake_images[:, 1, ..., tf.newaxis],
            train_generator_loss=train_generator_loss,
            train_discriminator_loss=train_discriminator_loss,
            train_real_inception_score=train_real_inception_score,
            train_fake_inception_score=train_fake_inception_score,
            train_frechet_inception_distance=train_frechet_inception_distance,
            valid_real_magnitude_spectrograms=valid_real_images[:, 0, ..., tf.newaxis],
            valid_real_instantaneous_frequencies=valid_real_images[:, 1, ..., tf.newaxis],
            valid_fake_magnitude_spectrograms=valid_fake_images[:, 0, ..., tf.newaxis],
            valid_fake_instantaneous_frequencies=valid_fake_images[:, 1, ..., tf.newaxis],
            valid_generator_loss=valid_generator_loss,
            valid_discriminator_loss=valid_discriminator_loss,
            valid_real_inception_score=valid_real_inception_score,
            valid_fake_inception_score=valid_fake_inception_score,
            valid_frechet_inception_distance=valid_frechet_inception_distance
        )

    def train(self, model_dir, total_steps, save_checkpoint_steps,
              save_train_summary_steps, save_valid_summary_steps,
              log_train_tensor_steps, log_valid_tensor_steps, config):

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
                    save_steps=save_train_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items() if "train" in name
                    ])
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_valid_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items() if "valid" in name
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if "train" in name and tensor.shape.ndims == 0
                    },
                    every_n_iter=log_train_tensor_steps,
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if "valid" in name and tensor.shape.ndims == 0
                    },
                    every_n_iter=log_valid_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                for name, operation in self.operations.items():
                    session.run(operation)
