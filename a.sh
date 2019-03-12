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

python main.py              --train  --generate \
                            --filenames nsynth-test/examples.tfrecord \
                            --sample_dir1 samples/log_mel_magnitude_spectrograms \
                            --sample_dir2 samples/mel_instantaneous_frequencies

python postprocess.py       samples/log_mel_magnitude_spectrograms \
                            samples/mel_instantaneous_frequencies \
                            samples/waveforms
