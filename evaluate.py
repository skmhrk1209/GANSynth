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
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    with open("pitch_counts.pickle", "rb") as file:
        pitch_counts = pickle.load(file)

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

        real_feature_batches = []
        real_logit_batches = [],
        fake_feature_batches = [],
        fake_logit_batches = []

        while True:
            try:
                real_feature_batch, real_logit_batch, fake_feature_batch, fake_logit_batch = session.run([
                    real_features, real_logits, fake_features, fake_logits
                ])
                real_feature_batches.append(real_feature_batch)
                real_logit_batches.append(real_logit_batch)
                fake_feature_batches.append(fake_feature_batch)
                fake_logit_batches.append(fake_logit_batch)
            except tf.errors.OutOfRangeError:
                break

        real_features = np.concatenate(real_feature_batches, axis=0)
        real_logits = np.concatenate(real_logit_batches, axis=0)
        fake_features = np.concatenate(fake_feature_batches, axis=0)
        fake_logits = np.concatenate(fake_logit_batches, axis=0)

        tf.logging.info("real_inception_score: {}, fake_inception_score: {}, frechet_inception_distance: {}".format(
            metrics.inception_score(
                logits=np.asanyarray(real_logits)[:, 1:]
            ),
            metrics.inception_score(
                logits=np.asanyarray(fake_logits)[:, 1:]
            ),
            metrics.frechet_inception_distance(
                real_features=np.asanyarray(real_features),
                fake_features=np.asanyarray(fake_features)
            )
        ))
