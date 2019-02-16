import tensorflow as tf
import numpy as np
import os
import itertools
import time
import cv2


def lerp(a, b, t):
    return a + (b - a) * t


class Model(object):

    class LossFunction:
        NS_GAN, WGAN = range(2)

    class GradientPenalty:
        ZERO_CENTERED, ONE_CENTERED = range(2)

    def __init__(self, dataset, generator, discriminator, loss_function,
                 gradient_penalty, hyper_params, name="gan", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            self.name = name
            self.dataset = dataset
            self.generator = generator
            self.discriminator = discriminator
            self.hyper_parameters = hyper_params

            self.batch_size = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="batch_size"
            )
            self.training = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name="training"
            )

            self.generator_global_step = tf.get_variable(
                name="generator_global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )
            self.discriminator_global_step = tf.get_variable(
                name="discriminator_global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )

            self.coloring_index = self.hyper_parameters.coloring_index_fn(
                global_step=tf.cast(self.discriminator_global_step, tf.float32)
            )

            self.next_reals = self.dataset.get_next()
            self.next_latents = tf.random_normal(
                shape=[self.batch_size, self.hyper_parameters.latent_size]
            )

            self.reals = tf.placeholder(
                dtype=tf.float32,
                shape=self.next_reals.shape,
                name="reals"
            )
            self.latents = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyper_parameters.latent_size],
                name="latents"
            )

            self.fakes = generator(
                inputs=self.latents,
                coloring_index=self.coloring_index,
                training=self.training,
                name="generator"
            )
            self.fakes = tf.identity(
                input=self.fakes,
                name="fakes"
            )

            self.real_logits = discriminator(
                inputs=self.reals,
                coloring_index=self.coloring_index,
                training=self.training,
                name="discriminator"
            )
            self.real_logits = tf.identity(
                input=self.real_logits,
                name="real_logits"
            )

            self.fake_logits = discriminator(
                inputs=self.fakes,
                coloring_index=self.coloring_index,
                training=self.training,
                name="discriminator",
                reuse=True
            )
            self.fake_logits = tf.identity(
                input=self.fake_logits,
                name="fake_logits"
            )

            #========================================================================#
            # two types of loss function
            # 1. NS-GAN loss function (https://arxiv.org/pdf/1406.2661.pdf)
            # 2. WGAN loss function (https://arxiv.org/pdf/1701.07875.pdf)
            #========================================================================#
            if loss_function == Model.LossFunction.NS_GAN:

                self.generator_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.fake_logits,
                        labels=tf.ones_like(self.fake_logits)
                    )
                )

                self.discriminator_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.real_logits,
                        labels=tf.ones_like(self.real_logits)
                    )
                )
                self.discriminator_loss += tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.fake_logits,
                        labels=tf.zeros_like(self.fake_logits)
                    )
                )

            elif loss_function == Model.LossFunction.WGAN:

                self.generator_loss = -tf.reduce_mean(self.fake_logits)

                self.discriminator_loss = -tf.reduce_mean(self.real_logits)
                self.discriminator_loss += tf.reduce_mean(self.fake_logits)

            else:
                raise ValueError("Invalid loss function")

            #========================================================================#
            # linear interpolation for gradient penalty
            #========================================================================#
            self.lerp_coefficients = tf.random_uniform(shape=[self.batch_size, 1, 1, 1])
            self.lerped = lerp(self.reals, self.fakes, self.lerp_coefficients)
            self.lerped_logits = discriminator(
                inputs=self.lerped,
                training=self.training,
                coloring_index=self.coloring_index,
                name="discriminator",
                reuse=True
            )
            
            #========================================================================#
            # two types of gradient penalty
            # 1. zero-centered gradient penalty (https://openreview.net/pdf?id=ByxPYjC5KQ)
            # -> NOT EFFECTIVE FOR NOW
            # 2. one-centered gradient penalty (https://arxiv.org/pdf/1704.00028.pdf)
            #
            # to avoid NaN exception, add epsilon inside sqrt()
            # (https://github.com/tdeboissiere/DeepLearningImplementations/issues/68)
            #========================================================================#
            self.gradients = tf.gradients(ys=self.lerped_logits, xs=self.lerped)[0]
            self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), axis=[1, 2, 3]) + 0.0001)

            if gradient_penalty == Model.GradientPenalty.ZERO_CENTERED:

                self.gradient_penalty = tf.reduce_mean(tf.square(self.slopes - 0.0))

            elif gradient_penalty == Model.GradientPenalty.ONE_CENTERED:

                self.gradient_penalty = tf.reduce_mean(tf.square(self.slopes - 1.0))

            else:
                raise ValueError("Invalid gradient penalty")

            self.discriminator_loss += self.gradient_penalty * self.hyper_parameters.gradient_coefficient

            self.generator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/generator".format(self.name)
            )
            self.discriminator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/discriminator".format(self.name)
            )

            self.generator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_parameters.learning_rate,
                beta1=self.hyper_parameters.beta1,
                beta2=self.hyper_parameters.beta2
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_parameters.learning_rate,
                beta1=self.hyper_parameters.beta1,
                beta2=self.hyper_parameters.beta2
            )

            #========================================================================#
            # to update moving_mean and moving_variance
            # for batch normalization when trainig,
            # run update operation before train operation
            # update operation is placed in tf.GraphKeys.UPDATE_OPS
            #========================================================================#
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                self.generator_train_op = self.generator_optimizer.minimize(
                    loss=self.generator_loss,
                    var_list=self.generator_variables,
                    global_step=self.generator_global_step
                )

                self.discriminator_train_op = self.discriminator_optimizer.minimize(
                    loss=self.discriminator_loss,
                    var_list=self.discriminator_variables,
                    global_step=self.discriminator_global_step
                )

            self.saver = tf.train.Saver()

            self.summary = tf.summary.merge([
                tf.summary.image("reals", self.reals, max_outputs=10),
                tf.summary.image("fakes", self.fakes, max_outputs=10),
                tf.summary.scalar("generator_loss", self.generator_loss),
                tf.summary.scalar("discriminator_loss", self.discriminator_loss),
                tf.summary.scalar("gradient_penalty", self.gradient_penalty),
            ])

    def initialize(self):

        session = tf.get_default_session()

        checkpoint = tf.train.latest_checkpoint(self.name)

        if checkpoint:
            self.saver.restore(session, checkpoint)
            print(checkpoint, "loaded")

        else:
            global_variables = tf.global_variables(scope=self.name)
            session.run(tf.variables_initializer(global_variables))
            print("global variables in {} initialized".format(self.name))

    def train(self, filenames, num_epochs, batch_size, buffer_size):

        session = tf.get_default_session()
        writer = tf.summary.FileWriter(self.name, session.graph)

        print("training started")

        start = time.time()

        self.dataset.initialize(
            filenames=filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size
        )

        feed_dict = {
            self.batch_size: batch_size,
            self.training: True
        }

        for i in itertools.count():

            try:
                reals, latents = session.run(
                    [self.next_reals, self.next_latents],
                    feed_dict=feed_dict
                )

            except tf.errors.OutOfRangeError:
                print("training ended")
                break

            feed_dict.update({
                self.reals: reals,
                self.latents: latents
            })

            session.run(
                [self.generator_train_op, self.discriminator_train_op],
                feed_dict=feed_dict
            )

            generator_global_step, discriminator_global_step = session.run(
                [self.generator_global_step, self.discriminator_global_step]
            )

            if generator_global_step % 100 == 0:

                generator_loss, discriminator_loss = session.run(
                    [self.generator_loss, self.discriminator_loss],
                    feed_dict=feed_dict
                )

                print("global_step: {}, generator_loss: {:.2f}".format(
                    generator_global_step,
                    generator_loss
                ))
                print("global_step: {}, discriminator_loss: {:.2f}".format(
                    discriminator_global_step,
                    discriminator_loss
                ))

                coloring_index = session.run(self.coloring_index)
                print("coloring_index: {:2f}".format(coloring_index))

                summary = session.run(self.summary, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=generator_global_step)

                if generator_global_step % 100000 == 0:

                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join(self.name, "model.ckpt"),
                        global_step=generator_global_step
                    )

                    tf.train.write_graph(
                        graph_or_graph_def=session.graph.as_graph_def(),
                        logdir=self.name,
                        name="graph.pb",
                        as_text=False
                    )

                    stop = time.time()
                    print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))
                    start = time.time()
