#!/bin/bash

#wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
#wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
#wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

#tar -xvf nsynth-train.jsonwav.tar.gz
#tar -xvf nsynth-valid.jsonwav.tar.gz
#tar -xvf nsynth-test.jsonwav.tar.gz

python preprocess.py
python make_tfrecord.py
python train.py --total_steps 100
python evaluate.py --total_steps 100
python generate.py --total_steps 100
python postprocess
