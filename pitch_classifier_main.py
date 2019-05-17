#=================================================================================================#
# TensorFlow implementation of GANSynth
#
# original paper
# [GANSynth: Adversarial Neural Audio Synthesis]
# (https://openreview.net/pdf?id=H1xQVn09FX)
#
# based on following papers
# [Progressive Growing of GANs for Improved Quality, Stability, and Variation]
# (https://arxiv.org/pdf/1710.10196.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
import functools
import argparse
import glob
from dataset import nsynth_input_fn
from models import PitchClassifier
from networks import ResNet
from utils import Dict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="pitch_classifier_model")
parser.add_argument('--filenames', type=str, default="nsynth*.tfrecord")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--total_steps", type=int, default=50000)
parser.add_argument('--train', action="store_true")
parser.add_argument('--evaluate', action="store_true")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    resnet = ResNet(
        conv_param=Dict(filters=64, kernel_size=[7, 7], strides=[2, 2]),
        pool_param=Dict(kernel_size=[3, 3], strides=[2, 2]),
        residual_params=[
            Dict(filters=64, strides=[1, 1], blocks=3),
            Dict(filters=128, strides=[2, 2], blocks=4),
            Dict(filters=256, strides=[2, 2], blocks=6),
            Dict(filters=512, strides=[2, 2], blocks=3)
        ],
        groups=32,
        classes=len(range(24, 85))
    )

    pitch_classifier = PitchClassifier(
        network=resnet,
        input_fn=functools.partial(
            nsynth_input_fn,
            filenames=glob.glob(args.filenames),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs if args.train else 1,
            shuffle=True if args.train else False,
            pitches=range(24, 85),
            sources=[0]
        ),
        spectral_params=Dict(
            waveform_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        ),
        # [Don't Decay the Learning Rate, Increase the Batch Size]
        # (https://arxiv.org/pdf/1711.00489.pdf)
        hyper_params=Dict(
            weight_decay=1e-4,
            learning_rate=lambda global_step: tf.train.exponential_decay(
                learning_rate=0.128 * args.batch_size / 256,
                global_step=global_step,
                decay_steps=70000 * args.num_epochs / 4 / args.batch_size,
                decay_rate=0.1
            ),
            momentum=0.9,
            use_nesterov=True
        )
    )

    config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=1.0
        )
    )

    if args.train:

        pitch_classifier.train(
            model_dir=args.model_dir,
            config=config,
            total_steps=args.total_steps,
            save_checkpoint_steps=1000,
            save_summary_steps=100,
            log_tensor_steps=100
        )

    if args.evaluate:

        pitch_classifier.evaluate(
            model_dir=args.model_dir,
            config=config
        )
