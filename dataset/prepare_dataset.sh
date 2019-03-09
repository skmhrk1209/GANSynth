# download tfrecord
#curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.tfrecord
#curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.tfrecord
curl -LO http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.tfrecord

# preprocess dataset
#python3 preprocess.py nsynth-train.tfrecord nsynth/train
#python3 preprocess.py nsynth-valid.tfrecord nsynth/valid
python3 preprocess.py nsynth-test.tfrecord nsynth/test

# make tfrecord
#python3 make_tfrecord.py nsynth/train/gt.txt nsynth_train.tfrecord
#python3 make_tfrecord.py nsynth/valid/gt.txt nsynth_valid.tfrecord
python3 make_tfrecord.py nsynth/test/gt.txt nsynth_test.tfrecord
