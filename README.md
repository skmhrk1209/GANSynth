## GANSynth: Adversarial Neural Audio Synthesis<br><i>TensorFlow implementation of the ICLR 2019 paper</i>

### Original paper 
* [GANSynth: Adversarial Neural Audio Synthesis](https://openreview.net/pdf?id=H1xQVn09FX)

### Based on following papers
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
* [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/pdf/1802.05957.pdf)
* [cGANs with Projection Discriminator](https://arxiv.org/pdf/1802.05637.pdf)

### Dataset
* [The NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)

### Architecture
* Progressive Training
* Spectral Normalization on generator and discriminator
* Conditional Batch Normalization on generator
* Conditional information projection on discriminator
