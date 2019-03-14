## GANSynth: Adversarial Neural Audio Synthesis<br><i>TensorFlow implementation of the ICLR 2019 paper</i>

### Original paper 
* [GANSynth: Adversarial Neural Audio Synthesis](https://openreview.net/pdf?id=H1xQVn09FX)

### Based on following papers
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

### Dataset
* [The NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)

### Usage
The original tfrecord file is very large, so it takes so long time to shuffle after each epoch.   
For better peformance, convert waveforms to spectrograms in advance and make tfrecord which contains the path to spectrograms and label.
```bash
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

tar -xvf nsynth-train.jsonwav.tar.gz
tar -xvf nsynth-valid.jsonwav.tar.gz
tar -xvf nsynth-test.jsonwav.tar.gz

python preprocess.py        nsynth-train/audio \
                            nsynth-train/log_mel_magnitude_spectrograms \
                            nsynth-train/mel_instantaneous_frequencies \

python preprocess.py        nsynth-valid/audio \
                            nsynth-valid/log_mel_magnitude_spectrograms \
                            nsynth-valid/mel_instantaneous_frequencies \

python preprocess.py        nsynth-test/audio \
                            nsynth-test/log_mel_magnitude_spectrograms \
                            nsynth-test/mel_instantaneous_frequencies \

python make_tfrecord.py     nsynth-train/log_mel_magnitude_spectrograms \
                            nsynth-train/mel_instantaneous_frequencies \
                            nsynth-train/examples.json \
                            nsynth-train/examples.tfrecord \

python make_tfrecord.py     nsynth-valid/log_mel_magnitude_spectrograms \
                            nsynth-valid/mel_instantaneous_frequencies \
                            nsynth-valid/examples.json \
                            nsynth-valid/examples.tfrecord \

python make_tfrecord.py     nsynth-test/log_mel_magnitude_spectrograms \
                            nsynth-test/mel_instantaneous_frequencies \
                            nsynth-test/examples.json \
                            nsynth-test/examples.tfrecord \

python train.py             --train_filenames nsynth-train/examples.tfrecord \
                            --valid_filenames nsynth-valid/examples.tfrecord \
                            --total_steps 1000000 \

python evaluate.py          --filenames nsynth-test/examples.tfrecord \
                            --total_steps 1000000 \

python generate.py          --sample_dir1 samples/log_mel_magnitude_spectrograms \
                            --sample_dir2 samples/mel_instantaneous_frequencies \
                            --total_steps 1000000 \

python postprocess.py       samples/log_mel_magnitude_spectrograms \
                            samples/mel_instantaneous_frequencies \
                            samples/waveforms \
```
