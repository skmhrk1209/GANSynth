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
import collections
import functools
import argparse
import glob
import json
import os
from scipy.io import wavfile
from dataset import nsynth_input_fn
from models import GANSynth
from networks import PGGAN
from utils import Dict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, default="nsynth*.tfrecord")
parser.add_argument('--classifier', type=str, default="pitch_classifier.pb")
parser.add_argument('--train', action="store_true")
parser.add_argument('--evaluate', action="store_true")
parser.add_argument('--generate', action="store_true")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def apply(function, dictionary):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            dictionary[key] = apply(function, value)
        dictionary = function(dictionary)
    return dictionary


with open("config.json") as file:
    config = json.load(file)
    config.update(vars(args))
    config = apply(Dict, config)
    print(f"config: {config}")

with tf.Graph().as_default():

    tf.set_random_seed(0)

    pggan = PGGAN(
        min_resolution=[2, 16],
        max_resolution=[128, 1024],
        min_channels=32,
        max_channels=256,
        growing_level=tf.cast(tf.divide(
            x=tf.train.create_global_step(),
            y=config.growing_steps
        ), tf.float32)
    )

    gan_synth = GANSynth(
        generator=pggan.generator,
        discriminator=pggan.discriminator,
        real_input_fn=functools.partial(
            nsynth_input_fn,
            filenames=glob.glob(config.filenames),
            batch_size=config.batch_size,
            num_epochs=config.num_epochs if config.train else 1,
            shuffle=True if config.train else False,
            pitches=range(24, 85),
            sources=[0]
        ),
        fake_input_fn=lambda: tf.random.normal([config.batch_size, 256]),
        spectral_params=config.spectral_params,
        hyper_params=config.hyper_params
    )

    session_config = tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=1.0
        )
    )

    if config.train:

        gan_synth.train(
            model_dir=config.model_dir,
            config=session_config,
            total_steps=config.total_steps,
            save_checkpoint_steps=1000,
            save_summary_steps=100,
            log_tensor_steps=100
        )

    if config.evaluate:

        with open(config.classifier, "rb") as file:
            classifier = tf.GraphDef.FromString(file.read())

        print(gan_synth.evaluate(
            model_dir=config.model_dir,
            config=session_config,
            classifier=classifier,
            input_name="images:0",
            output_names=["features:0", "logits:0"]
        ))

    if config.generate:

        os.makedirs("samples", exist_ok=True)

        generator = gan_synth.generate(
            model_dir=config.model_dir,
            config=session_config
        )

        num_waveforms = 0
        for waveforms in generator:
            for waveform in waveforms:
                wavfile.write(f"samples/{num_waveforms}.wav", rate=16000, data=waveform)
                num_waveforms += 1

        print(f"{num_waveforms} waveforms are generated in `samples` directory")
