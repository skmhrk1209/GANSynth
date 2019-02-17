import tensorflow as tf
import numpy as np
import os
import itertools
import time
import pitch


class Model(object):

    def __init__(self, generator, discriminator, input_fn,
                 hyper_params, name="gan_synth", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            self.name = name
            self.input_fn = input_fn
            self.generator = generator
            self.discriminator = discriminator
            self.hyper_params = hyper_params
            # =========================================================================================
            # parameters
            self.batch_size = tf.placeholder(
                dtype=tf.int32,
                shape=[]
            )
            self.training = tf.placeholder(
                dtype=tf.bool,
                shape=[]
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
            self.coloring_index = self.hyper_params.coloring_index_fn(
                global_step=tf.cast(self.discriminator_global_step, tf.float32)
            )
            # =========================================================================================
            # input_fn for real data
            self.next_real_images, self.next_real_labels = self.input_fn()
            # =========================================================================================
            # input_fn for fake data
            self.next_fake_labels = tf.one_hot(
                indices=tf.reshape(
                    tensor=tf.multinomial(
                        logits=tf.log([tf.cast([
                            count for pitch, count
                            in sorted(pitch.counts.items())
                        ], tf.float32)]),
                        num_samples=self.batch_size
                    ),
                    shape=[self.batch_size]
                ),
                depth=len(pitch.counts)
            )
            self.next_fake_latents = tf.random_normal(
                shape=[self.batch_size, self.hyper_params.latent_size]
            )
            # =========================================================================================
            # placeholders for real data
            self.real_images = tf.placeholder(
                dtype=tf.float32,
                shape=self.next_real_images.shape
            )
            self.real_labels = tf.placeholder(
                dtype=tf.float32,
                shape=self.next_real_labels.shape
            )
            # =========================================================================================
            # placeholders for fake data
            self.fake_latents = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.hyper_params.latent_size]
            )
            self.fake_images = generator(
                inputs=self.latents,
                coloring_index=self.coloring_index,
                training=self.training
            )
            self.fake_labels = tf.placeholder(
                dtype=tf.float32,
                shape=self.next_fake_labels.shape
            )
            # =========================================================================================
            # logits for real data
            self.real_logits = discriminator(
                inputs=self.real_images,
                conditions=self.real_labels,
                coloring_index=self.coloring_index,
                training=self.training,
                name="discriminator"
            )
            # =========================================================================================
            # logits for fake data
            self.fake_logits = discriminator(
                inputs=self.fake_images,
                conditions=self.fake_labels,
                coloring_index=self.coloring_index,
                training=self.training,
                name="discriminator",
                reuse=True
            )
            #========================================================================#
            # hinge loss for generator and discriminator
            self.discriminator_loss = tf.reduce_mean(tf.nn.relu(1 - self.real_logits))
            self.discriminator_loss += tf.reduce_mean(tf.nn.relu(1 + self.fake_logits))
            self.generator_loss = -tf.reduce_mean(self.fake_logits)
            #========================================================================#
            # variables for generator and discriminator
            self.generator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/generator".format(self.name)
            )
            self.discriminator_variables = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES,
                scope="{}/discriminator".format(self.name)
            )
            #========================================================================#
            # optimizer for generator and discriminator
            self.generator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_params.generator_learning_rate,
                beta1=self.hyper_params.generator_beta1,
                beta2=self.hyper_params.generator_beta2
            )
            self.discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_params.discriminator_learning_rate,
                beta1=self.hyper_params.discriminator_beta1,
                beta2=self.hyper_params.discriminator_beta2
            )
            #========================================================================#
            # training op for generator and discriminator
            # to update moving_mean and moving_variance
            # for batch normalization when trainig,
            # run update operation before train operation
            # update operation is placed in tf.GraphKeys.UPDATE_OPS
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
            #========================================================================#
            # utilities
            self.saver = tf.train.Saver()
            self.summary = tf.summary.merge([
                tf.summary.image("reals", self.real_images, max_outputs=2),
                tf.summary.image("fakes", self.fake_images, max_outputs=2),
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

        session.run(
            fetches=tf.tables_initializer(),
            feed_dict={
                self.filenames: filenames,
                self.num_epochs: num_epochs,
                self.batch_size: batch_size,
                self.buffer_size: buffer_size
            }
        )
        print("dataset iterator initialized")

        print("training started")
        start = time.time()

        feed_dict = {self.batch_size: batch_size, self.training: True}

        for i in itertools.count():

            try:
                real_images, real_labels = session.run(
                    fetches=[self.next_real_images, self.next_real_labels],
                    feed_dict=feed_dict
                )
                fake_latents, fake_labels = session.run(
                    fetches=[self.next_fake_latents, self.next_fake_labels],
                    feed_dict=feed_dict
                )

            except tf.errors.OutOfRangeError:
                print("training ended")
                break

            feed_dict.update({
                self.real_images: real_images,
                self.real_labels: real_labels,
                self.fake_latents: fake_latents,
                self.fake_labels: fake_labels
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

                    stop = time.time()
                    print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))
                    start = time.time()
