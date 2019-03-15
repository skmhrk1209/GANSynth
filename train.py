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
import argparse
import functools
import pickle
from dataset import nsynth_input_fn
from model import GANSynth
from network import PGGAN
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--train_filenames', type=str, nargs="+", default=["nsynth_train_examples.tfrecord"])
parser.add_argument('--valid_filenames', type=str, nargs="+", default=["nsynth_valid_examples.tfrecord"])
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--valid_batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=None)
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

    gan_synth = GANSynth(
        generator=pggan.generator,
        discriminator=pggan.discriminator,
        train_real_input_fn=functools.partial(
            nsynth_input_fn,
            filenames=args.train_filenames,
            batch_size=args.train_batch_size,
            num_epochs=args.num_epochs,
            shuffle=True,
            pitches=pitch_counts.keys()
        ),
        train_fake_input_fn=lambda: (
            tf.random_normal([args.train_batch_size, 256]),
            tf.one_hot(tf.reshape(tf.multinomial(
                logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
                num_samples=args.train_batch_size
            ), [args.train_batch_size]), len(pitch_counts))
        ),
        valid_real_input_fn=functools.partial(
            nsynth_input_fn,
            filenames=args.valid_filenames,
            batch_size=args.valid_batch_size,
            num_epochs=args.num_epochs,
            shuffle=True,
            pitches=pitch_counts.keys()
        ),
        valid_fake_input_fn=lambda: (
            tf.random_normal([args.valid_batch_size, 256]),
            tf.one_hot(tf.reshape(tf.multinomial(
                logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
                num_samples=args.valid_batch_size
            ), [args.valid_batch_size]), len(pitch_counts))
        ),
        hyper_params=Struct(
            generator_learning_rate=8e-4,
            generator_beta1=0.0,
            generator_beta2=0.99,
            discriminator_learning_rate=8e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            gradient_penalty_weight=10.0,
            generator_classification_weight=10.0,
            discriminator_classification_weight=10.0,
        )
    )
    gan_synth.train(
        model_dir=args.model_dir,
        total_steps=args.total_steps,
        save_checkpoint_steps=1000,
        save_summary_steps=100,
        log_step_count_steps=100,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            )
        )
    )
