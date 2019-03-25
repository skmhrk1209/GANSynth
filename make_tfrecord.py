import tensorflow as tf
import itertools
import random
import json
import sys
from pathlib import Path


def main(in_file, out_file):

    with tf.io.TFRecordWriter(out_file) as writer:

        with open(in_file) as file:
            ground_truth = json.load(file)

        for key, value in ground_truth.items():
            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature=dict(
                            path_to_magnitude_spectrogram=tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[value["path_to_magnitude_spectrogram"].encode()]
                                )
                            ),
                            path_to_instantaneous_frequency=tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[value["path_to_instantaneous_frequency"].encode()]
                                )
                            ),
                            instrument_source=tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[value["instrument_source"]]
                                )
                            ),
                            pitch=tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[value["pitch"]]
                                )
                            ),
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
            path_to_waveform="nsynth-train/audio/{}.wav".format(key),
            path_to_magnitude_spectrogram="nsynth-train/magnitude_spectrograms/{}.jpg".format(key),
            path_to_instantaneous_frequency="nsynth-train/instantaneous_frequencies/{}.jpg".format(key)
        ))

    for key in nsynth_valid_examples:
        nsynth_valid_examples[key].update(dict(
            path_to_waveform="nsynth-valid/audio/{}.wav".format(key),
            path_to_magnitude_spectrogram="nsynth-valid/magnitude_spectrograms/{}.jpg".format(key),
            path_to_instantaneous_frequency="nsynth-valid/instantaneous_frequencies/{}.jpg".format(key)
        ))

    for key in nsynth_test_examples:
        nsynth_test_examples[key].update(dict(
            path_to_waveform="nsynth-test/audio/{}.wav".format(key),
            path_to_magnitude_spectrogram="nsynth-test/magnitude_spectrograms/{}.jpg".format(key),
            path_to_instantaneous_frequency="nsynth-test/instantaneous_frequencies/{}.jpg".format(key)
        ))

    nsynth_examples = list(dict(
        **nsynth_train_examples,
        **nsynth_valid_examples,
        **nsynth_test_examples
    ).items())

    random.shuffle(nsynth_examples)

    nsynth_train_examples, nsynth_valid_examples, nsynth_test_examples = [
        nsynth_examples[begin:end] for begin, end in zip(
            [None, int(len(nsynth_examples) * 0.8), int(len(nsynth_examples) * 0.9)],
            [int(len(nsynth_examples) * 0.8), int(len(nsynth_examples) * 0.9), None]
        )
    ]

    with open("nsynth_train_examples.json", "w") as file:
        json.dump(dict(nsynth_train_examples), file, indent=4)

    with open("nsynth_valid_examples.json", "w") as file:
        json.dump(dict(nsynth_valid_examples), file, indent=4)

    with open("nsynth_test_examples.json", "w") as file:
        json.dump(dict(nsynth_test_examples), file, indent=4)

    main("nsynth_train_examples.json", "nsynth_train_examples.tfrecord")
    main("nsynth_valid_examples.json", "nsynth_valid_examples.tfrecord")
    main("nsynth_test_examples.json", "nsynth_test_examples.tfrecord")
