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
#
# [Spectral Normalization for Generative Adversarial Networks]
# (https://arxiv.org/pdf/1802.05957.pdf)
#
# [cGANs with Projection Discriminator]
# (https://arxiv.org/pdf/1802.05637.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
import argparse
import functools
import itertools
import pickle
from dataset import NSynth
from model import GANSynth
from residual_network import PGGAN
from attrdict import AttrDict as Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=32)
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
        max_channels=256,
        scale_weight=False,
        apply_spectral_norm=True
    )

    nsynth = NSynth(
        pitch_counts=pitch_counts,
        audio_length=64000,
        sample_rate=16000,
        spectrogram_shape=[128, 1024],
        overlap=0.75
    )

    gan_synth = GANSynth(
        discriminator=pggan.discriminator,
        generator=pggan.generator,
        real_input_fn=functools.partial(
            nsynth.real_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True
        ),
        fake_input_fn=functools.partial(
            nsynth.fake_input_fn,
            latent_size=256,
            batch_size=args.batch_size
        ),
        hyper_params=Param(
            discriminator_learning_rate=2e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.9,
            generator_learning_rate=1e-4,
            generator_beta1=0.0,
            generator_beta2=0.9
        ),
        name=args.model_dir
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    with tf.Session(config=config) as session:

        gan_synth.initialize()
        gan_synth.train(
            progress_steps=args.total_steps // 2,
            total_steps=args.total_steps,
        )
