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
import skimage
import argparse
import pickle
import metrics
from dataset import nsynth_input_fn
from network import PGGAN
from utils import Struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument("--sample_dir1", type=str, default="samples/log_mel_magnitude_spectrograms")
parser.add_argument("--sample_dir2", type=str, default="samples/mel_instantaneous_frequencies")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth-test/examples.tfrecord"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--total_steps", type=int, default=1000000)
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

    images = pggan.generator(
        latents=tf.random_normal([args.batch_size, 256]),
        labels=tf.one_hot(tf.reshape(tf.multinomial(
            logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
            num_samples=args.batch_size
        ), [args.batch_size]), len(pitch_counts))
    )

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

        sample_dir1 = Path(args.sample_dir1)
        sample_dir2 = Path(args.sample_dir1)

        if not sample_dir1.exists():
            sample_dir1.mkdir()
        if not sample_dir2.exists():
            sample_dir2.mkdir()

        def linear_map(inputs, in_min, in_max, out_min, out_max):
            return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)

        for image in session.run(images):
            skimage.io.imsave(
                fname=sample_dir1 / "{}.jpg".format(len(list(sample_dir1.glob("*.jpg")))),
                arr=linear_map(image[0], -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
            )
            skimage.io.imsave(
                fname=sample_dir2 / "{}.jpg".format(len(list(sample_dir2.glob("*.jpg")))),
                arr=linear_map(image[1], -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
            )
