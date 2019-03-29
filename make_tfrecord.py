import tensorflow as tf
import random
import json


def main(filename, examples):

    with tf.io.TFRecordWriter(filename) as writer:

        for key, value in examples.items():
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


if __name__ == "__main__":

    with open("nsynth-train/examples.json") as file:
        nsynth_train_examples = json.load(file)

    with open("nsynth-valid/examples.json") as file:
        nsynth_valid_examples = json.load(file)

    with open("nsynth-test/examples.json") as file:
        nsynth_test_examples = json.load(file)

    for key in nsynth_train_examples:
        nsynth_train_examples[key].update(dict(
            path="nsynth-train/audio/{}.wav".format(key)
        ))

    for key in nsynth_valid_examples:
        nsynth_valid_examples[key].update(dict(
            path="nsynth-valid/audio/{}.wav".format(key)
        ))

    for key in nsynth_test_examples:
        nsynth_test_examples[key].update(dict(
            path="nsynth-test/audio/{}.wav".format(key)
        ))

    nsynth_examples = list(dict(
        **nsynth_train_examples,
        **nsynth_valid_examples,
        **nsynth_test_examples
    ).items())

    random.shuffle(nsynth_examples)

    nsynth_train_examples, nsynth_valid_examples, nsynth_test_examples = [
        dict(nsynth_examples[begin:end]) for begin, end in zip(
            [None, int(len(nsynth_examples) * 0.8), int(len(nsynth_examples) * 0.9)],
            [int(len(nsynth_examples) * 0.8), int(len(nsynth_examples) * 0.9), None]
        )
    ]

    main(nsynth_train_examples, "nsynth_train.tfrecord")
    main(nsynth_valid_examples, "nsynth_valid.tfrecord")
    main(nsynth_test_examples, "nsynth_test.tfrecord")
