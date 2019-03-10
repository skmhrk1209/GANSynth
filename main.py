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
import nsynth
from model import GANSynth
from network import PGGAN
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument("--sample_dir1", type=str, default="samples/log_mel_magnitude_spectrograms")
parser.add_argument("--sample_dir2", type=str, default="samples/mel_instantaneous_frequencies")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth-train/examples.tfrecord"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with open("pitch_counts.pickle", "rb") as file:
    pitch_counts = pickle.load(file)

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

    gan_synth = GANSynth(
        generator=pggan.generator,
        discriminator=pggan.discriminator,
        real_input_fn=functools.partial(
            nsynth.input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True,
            pitches=pitch_counts.keys()
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 256]),
            tf.one_hot(tf.reshape(tf.multinomial(
                logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
                num_samples=args.batch_size
            ), [args.batch_size]), len(pitch_counts))
        ),
        hyper_params=Struct(
            generator_learning_rate=8e-4,
            generator_beta1=0.0,
            generator_beta2=0.99,
            discriminator_learning_rate=8e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            real_zero_centered_gradient_penalty_weight=10.0,
            fake_zero_centered_gradient_penalty_weight=0.0,
            one_centered_gradient_penalty_weight=10.0,
            generator_auxiliary_classification_weight=10.0,
            discriminator_auxiliary_classification_weight=10.0,
        )
    )

    if args.train:

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

    if args.generate:

        gan_synth.generate(
            model_dir=args.model_dir,
            sample_dir1=args.sample_dir1,
            sample_dir2=args.sample_dir2,
            steps=args.steps,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
