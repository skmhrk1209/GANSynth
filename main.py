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
import functools
import pickle
from dataset import NSynth
from model import GAN
from network import PGGAN
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with open("pitch_counts.pickle", "rb") as file:
    pitch_counts = pickle.load(file)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    pggan = PGGAN(
        min_resolution=[1, 8],
        max_resolution=[128, 1024],
        min_channels=16,
        max_channels=512,
        growing_level=tf.cast(tf.get_variable(
            name="global_step",
            initializer=0,
            trainable=False
        ) / args.total_steps, tf.float32)
    )

    nsynth = NSynth(
        pitch_counts=pitch_counts,
        audio_length=64000,
        sample_rate=16000,
        spectrogram_shape=[128, 1024],
        overlap=0.75
    )

    gan = GAN(
        discriminator=pggan.discriminator,
        generator=pggan.generator,
        real_input_fn=functools.partial(
            nsynth.input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 512]),
            tf.one_hot(tf.reshape(tf.random.multinomial(
                logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
                num_samples=args.batch_size
            ), [args.batch_size]), len(pitch_counts))
        ),
        hyper_params=Param(
            discriminator_learning_rate=8e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.9,
            generator_learning_rate=8e-4,
            generator_beta1=0.0,
            generator_beta2=0.9
        ),
        model_dir=args.model_dir
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    with tf.Session(config=config) as session:

        gan.initialize()
        gan.train(args.total_steps)
