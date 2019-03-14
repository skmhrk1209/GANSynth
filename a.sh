#!/bin/bash

wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

tar -xvf nsynth-test.jsonwav.tar.gz

python preprocess.py        nsynth-test/audio \
                            nsynth-test/log_mel_magnitude_spectrograms \
                            nsynth-test/mel_instantaneous_frequencies

python make_tfrecord.py     nsynth-test/log_mel_magnitude_spectrograms \
                            nsynth-test/mel_instantaneous_frequencies \
                            nsynth-test/examples.json \
                            nsynth-test/examples.tfrecord

python main.py              --evaluate \
                            --filenames nsynth-test/examples.tfrecord \
