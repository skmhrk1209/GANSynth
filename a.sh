#!/bin/bash

wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz

tar -xvf nsynth-valid.jsonwav.tar.gz

python preprocess.py        nsynth-valid/audio \
                            nsynth-valid/log_mel_magnitude_spectrograms \
                            nsynth-valid/mel_instantaneous_frequencies

python make_tfrecord.py     nsynth-valid/log_mel_magnitude_spectrograms \
                            nsynth-valid/mel_instantaneous_frequencies \
                            nsynth-valid/examples.json \
                            nsynth-valid/examples.tfrecord
