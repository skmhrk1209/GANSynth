import tensorflow as tf
import numpy as np
import itertools
import functools
import os


class GAN(object):

    def __init__(self, discriminator, generator, real_input_fn, fake_input_fn, hyper_params, model_dir):

        # =========================================================================================
        # input_fn for real data and fake data
        self.real_images, self.real_labels = real_input_fn()
        self.latents, self.fake_labels = fake_input_fn()
        # =========================================================================================
        # generated fake data
        self.fake_images = generator(
            latents=self.latents,
            labels=self.fake_labels,
            name="generator"
        )
        # =========================================================================================
        # logits for real data and fake data
        self.real_logits = discriminator(
            images=self.real_images,
            labels=self.real_labels,
            name="discriminator"
        )
        self.fake_logits = discriminator(
            images=self.fake_images,
            labels=self.fake_labels,
            name="discriminator",
            reuse=True
        )
        #========================================================================#
        # loss functions from
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)

        self.discriminator_loss = tf.nn.softplus(self.fake_logits)
        self.discriminator_loss += tf.nn.softplus(-self.real_logits)

        # zero-centerd gradient penalty
        if hyper_params.r1_gamma:
            real_loss = tf.reduce_sum(self.real_logits)
            real_grads = tf.gradients(real_loss, [self.real_images])[0]
            r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
            self.discriminator_loss += 0.5 * hyper_params.r1_gamma * r1_penalty

        # zero-centerd gradient penalty
        if hyper_params.r2_gamma:
            fake_loss = tf.reduce_sum(self.fake_logits)
            fake_grads = tf.gradients(fake_loss, [self.fake_images])[0]
            r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3])
            self.discriminator_loss += 0.5 * hyper_params.r2_gamma * r2_penalty

        self.generator_loss = tf.nn.softplus(-self.fake_logits)
        #========================================================================#
        # variables for discriminator and generator
        self.discriminator_variables = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="discriminator"
        )
        self.generator_variables = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="generator"
        )
        #========================================================================#
        # optimizer for discriminator and generator
        self.discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )
        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        #========================================================================#
        # global_step
        with tf.variable_scope("", reuse=True):
            self.global_step = tf.get_variable(name="global_step", dtype=tf.int32)
        #========================================================================#
        # training op for generator and discriminator
        self.discriminator_train_op = self.discriminator_optimizer.minimize(
            loss=self.discriminator_loss,
            var_list=self.discriminator_variables
        )
        self.generator_train_op = self.generator_optimizer.minimize(
            loss=self.generator_loss,
            var_list=self.generator_variables,
            global_step=self.global_step
        )
        #========================================================================#
        # update ops for discriminator and generator
        # NOTE: tf.control_dependencies doesn't work
        self.discriminator_update_ops = tf.get_collection(
            key=tf.GraphKeys.UPDATE_OPS,
            scope="discriminator"
        )
        self.generator_update_ops = tf.get_collection(
            key=tf.GraphKeys.UPDATE_OPS,
            scope="generator"
        )
        self.discriminator_train_op = tf.group([self.discriminator_train_op, self.discriminator_update_ops])
        self.generator_train_op = tf.group([self.generator_train_op, self.generator_update_ops])
        #========================================================================#
        # utilities
        self.model_dir = model_dir
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge([
            tf.summary.image("real_images", tf.transpose(self.real_images, [0, 2, 3, 1]), max_outputs=2),
            tf.summary.image("fake_images", tf.transpose(self.fake_images, [0, 2, 3, 1]), max_outputs=2),
            tf.summary.scalar("discriminator_loss", self.discriminator_loss),
            tf.summary.scalar("generator_loss", self.generator_loss)
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
                    global_step. discriminator_loss, generator_loss
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
