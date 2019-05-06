import tensorflow as tf
import pathlib
import random
import json


if __name__ == "__main__":

    nsynth_all_examples = {}
    for filename in pathlib.Path(".").glob("nsynth*/*.json"):
        with open(filename) as file:
            nsynth_examples = json.load(file)
            for key, value in nsynth_examples.items():
                value.update(dict(path=str(filename.parent/"audio"/f"{key}.wav")))
            nsynth_all_examples.update(nsynth_examples)

    nsynth_all_examples = list(nsynth_all_examples.items())
    random.shuffle(nsynth_all_examples)

    nsynth_train_examples = nsynth_all_examples[:int(len(nsynth_all_examples) * 0.8)]
    nsynth_test_examples = nsynth_all_examples[int(len(nsynth_all_examples) * 0.8):]

    for nsynth_name, nsynth_examples in [("nsynth_train", nsynth_train_examples), ("nsynth_test", nsynth_test_examples)]:
        with tf.io.TFRecordWriter(f"{nsynth_name}.tfrecord") as writer:
            for key, value in nsynth_examples:
                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(
                                path=tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[value["path"].encode()]
                                    )
                                ),
                                pitch=tf.train.Feature(
                                    int64_list=tf.train.Int64List(
                                        value=[value["pitch"]]
                                    )
                                ),
                                source=tf.train.Feature(
                                    int64_list=tf.train.Int64List(
                                        value=[value["instrument_source"]]
                                    )
                                )
                            )
                        )
                    ).SerializeToString()
                )
