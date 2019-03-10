## GANSynth: Adversarial Neural Audio Synthesis<br><i>TensorFlow implementation of the ICLR 2019 paper</i>

### Original paper 
* [GANSynth: Adversarial Neural Audio Synthesis](https://openreview.net/pdf?id=H1xQVn09FX)

### Based on following papers
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)

### Dataset
* [The NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)

### Usage
[](The original tfrecord file is very large, so it takes so long time to shuffle after each epoch. For better peformance, it's better to convert waveforms to spectrograms in advance and make tfrecord which contains the path to spectrograms and label.)

```bash
> curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz

> tar -xvf nsynth-train.jsonwav.tar.gz

> python preprocess.py      nsynth-train/audio \
                            nsynth-train/log_mel_magnitude_spectrograms \
                            nsynth-train/mel_instantaneous_frequencies

> python make_tfrecord.py   nsynth-train/log_mel_magnitude_spectrograms \
                            nsynth-train/mel_instantaneous_frequencies \
                            nsynth-train/examples.json \
                            nsynth-train/examples.tfrecord

> python main.py            --train  --generate \
                            --filenames nsynth-train/examples.tfrecord \
                            --sample_dir1 samples/log_mel_magnitude_spectrograms \
                            --sample_dir2 samples/mel_instantaneous_frequencies

> python postprocess.py     samples/log_mel_magnitude_spectrograms \
                            samples/mel_instantaneous_frequencies \
                            samples/waveforms
```