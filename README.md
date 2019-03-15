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

python preprocess.py
python make_tfrecord.py
python train.py
python evaluate.py
python generate.py
python postprocess
```
