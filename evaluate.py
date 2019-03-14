#=================================================================================================#
# TensorFlow implementation of GANSynth
#
# original paper
# [GANSynth: Adversarial Neural Audio Synthesis]
# (https://openreview.net/pdf?id=H1xQVn09FX)
#
# based on following papers
#
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
import argparse
import pickle
import metrics
from dataset import nsynth_input_fn
from network import PGGAN
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth-test/examples.tfrecord"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    pggan = PGGAN(
        min_resolution=[2, 16],
        max_resolution=[128, 1024],
        min_channels=32,
        max_channels=256,
        growing_level=tf.cast(tf.divide(
            x=tf.train.create_global_step(),
            y=args.total_steps
        ), tf.float32)
    )

    with open("pitch_counts.pickle", "rb") as file:
        pitch_counts = pickle.load(file)

    real_images, real_labels = nsynth_input_fn(
        filenames=args.filenames,
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=False,
        pitches=pitch_counts.keys()
    )

    fake_latents, fake_labels = (
        tf.random_normal([args.batch_size, 256]),
        tf.one_hot(tf.reshape(tf.multinomial(
            logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
            num_samples=args.batch_size
        ), [args.batch_size]), len(pitch_counts))
    )

    fake_images = pggan.generator(fake_latents, fake_labels)
    real_features, real_logits = pggan.discriminator(real_images, real_labels)
    fake_features, fake_logits = pggan.discriminator(fake_images, fake_labels)

    with tf.train.SingularMonitoredSession(
        scaffold=tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.group(
                tf.local_variables_initializer(),
                tf.tables_initializer()
            )
        ),
        checkpoint_dir=args.model_dir,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            )
        )
    ) as session:

        def generator():
            while True:
                try:
                    yield session.run([real_features, real_logits, fake_features, fake_logits])
                except tf.errors.OutOfRangeError:
                    break

        real_features, real_logits, fake_features, fake_logits = map(np.concatenate, zip(*generator()))

        tf.logging.info("real_inception_score: {}, fake_inception_score: {}, frechet_inception_distance: {}".format(
            metrics.inception_score(real_logits[:, 1:]),
            metrics.inception_score(fake_logits[:, 1:]),
            metrics.frechet_inception_distance(real_features, fake_features)
        ))
