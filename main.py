#=================================================================================================#
# Implementation of Progressive Growing GAN
#
# 2018/10/01 Hiroki Sakuma
# (https://github.com/skmhrk1209/PGGAN)
#
# original paper
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#
# tuned up as described by
# [Are GANs Created Equal? A Large-Scale Study]
# (https://arxiv.org/pdf/1711.10337.pdf)
# [The GAN Landscape: Losses, Architectures, Regularization, and Normalization]
# (https://arxiv.org/pdf/1807.04720.pdf)
#=================================================================================================#

import tensorflow as tf
import argparse
import dataset
import functools
import itertools
import pitch
from models import gan
from networks import dcgan
from attrdict import AttrDict as Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth_test.tfrecord"], help="tfrecords")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--max_steps", type=int, default=100000, help="maximum number of training steps")
parser.add_argument("--steps", type=int, default=None, help="number of test steps")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

pitches = np.load("pitches.npy")
counts = np.load("counts.npy")

gan_model = gan.Model(
    generator=dcgan.Generator(
        min_resolution=[4, 8],
        max_resolution=[256, 512],
        min_filters=16,
        max_filters=1024,
        data_format=args.data_format,
    ),
    discriminator=dcgan.Discriminator(
        min_resolution=[4, 8],
        max_resolution=[256, 512],
        min_filters=16,
        max_filters=1024,
        data_format=args.data_format
    ),
    real_input_fn=functools.partial(
        dataset.input_fn,
        filenames=args.filenames,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True,
        audio_length=64000,
        pitches=pitches,
        spectrogram_shape=[256, 512],
        overlap=0.75,
        sample_rate=16000,
        mel_downscale=1
    ),
    fake_input_fn=lambda: (
        tf.one_hot(
            indices=tf.reshape(
                tensor=tf.multinomial(
                    logits=tf.log([tf.cast(counts, tf.float32)]),
                    num_samples=args.batch_size
                ),
                shape=[args.batch_size]
            ),
            depth=len(pitches)
        ),
        tf.random_normal(
            shape=[args.batch_size, 128]
        )
    ),
    hyper_params=Param(
        generator_learning_rate=0.0002,
        generator_beta1=0.5,
        generator_beta2=0.999,
        discriminator_learning_rate=0.0002,
        discriminator_beta1=0.5,
        discriminator_beta2=0.999,
        coloring_index_fn=lambda global_step: global_step / 10000.0 + 1.0
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

    gan_model.initialize()

    if args.train:

        gan_model.train(args.max_steps)
