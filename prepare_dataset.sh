#!/bin/bash

wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz

tar -xvf nsynth-train.jsonwav.tar.gz

python preprocess.py        nsynth-train/audio \
                            nsynth-train/log_mel_magnitude_spectrograms \
                            nsynth-train/mel_instantaneous_frequencies

python make_tfrecord.py     nsynth-train/log_mel_magnitude_spectrograms \
                            nsynth-train/mel_instantaneous_frequencies \
                            nsynth-train/examples.json \
                            nsynth-train/examples.tfrecord

python main.py              --train  --generate \
                            --filenames nsynth-train/examples.tfrecord \
                            --sample_dir1 samples/log_mel_magnitude_spectrograms \
                            --sample_dir2 samples/mel_instantaneous_frequencies

python postprocess.py       samples/log_mel_magnitude_spectrograms \
                            samples/mel_instantaneous_frequencies \
                            samples/waveforms
