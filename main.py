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
from dataset import nsynth_fake_input_fn
from model import GANSynth
from network import PGGAN
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, nargs="+")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument('--train', action="store_true")
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--generate', action="store_true")
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

    gan_synth = GANSynth(
        generator=pggan.generator,
        discriminator=pggan.discriminator,
        real_input_fn=functools.partial(
            nsynth_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs if args.train else 1,
            shuffle=True if args.train else False,
            pitches=list(range(24, 85))
        ),
        fake_input_fn=lambda: (
            tf.random.normal([args.batch_size, 256])
        ),
        hyper_params=Struct(
            generator_learning_rate=8e-4,
            generator_beta1=0.0,
            generator_beta2=0.99,
            discriminator_learning_rate=8e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            mode_seeking_loss_weight=1.0,
            real_gradient_penalty_weight=5.0,
            fake_gradient_penalty_weight=0.0,
        )
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    if args.train:
        gan_synth.train(
            model_dir=args.model_dir,
            total_steps=args.total_steps,
            save_checkpoint_steps=1000,
            save_summary_steps=100,
            log_tensor_steps=100,
            config=config
        )

    if args.evaluate:
        gan_synth.evaluate(
            model_dir=args.model_dir,
            config=config
        )

    if args.generate:
        gan_synth.generate(
            model_dir=args.model_dir,
            config=config
        )
