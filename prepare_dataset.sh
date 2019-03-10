# curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
# curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

# tar -xvf nsynth-train.jsonwav.tar.gz
# tar -xvf nsynth-valid.jsonwav.tar.gz
tar -xvf nsynth-test.jsonwav.tar.gz

# python preprocess.py nsynth-train/audio nsynth-train/log_mel_magnitude_spectrograms nsynth-train/mel_instantaneous_frequencies
# python preprocess.py nsynth-valid/audio nsynth-valid/log_mel_magnitude_spectrograms nsynth-valid/mel_instantaneous_frequencies
python preprocess.py        nsynth-test/audio \
                            nsynth-test/log_mel_magnitude_spectrograms \
                            nsynth-test/mel_instantaneous_frequencies

#python make_tfrecord.py nsynth/train/gt.txt nsynth_train.tfrecord
#python make_tfrecord.py nsynth/valid/gt.txt nsynth_valid.tfrecord
python make_tfrecord.py     nsynth-test/log_mel_magnitude_spectrograms \
                            nsynth-test/mel_instantaneous_frequencies \
                            nsynth-test/examples.json \
                            nsynth-test/examples.tfrecord

python main.py --generate --filenames nsynth-test/examples.tfrecord --total_steps 1

python postprocess.py       samples/log_mel_magnitude_spectrograms \
                            samples/mel_instantaneous_frequencies \
                            samples/waveforms