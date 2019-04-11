import tensorflow as tf
import pathlib
import json


if __name__ == "__main__":

    with tf.io.TFRecordWriter("nsynth.tfrecord") as writer:
        for filename in pathlib.Path(".").glob("nsynth*/*.json"):
            with open(filename) as file:
                for key, value in json.load(file).items():
                    writer.write(
                        record=tf.train.Example(
                            features=tf.train.Features(
                                feature=dict(
                                    path=tf.train.Feature(
                                        bytes_list=tf.train.BytesList(
                                            value=[str(filename.parent/"audio"/f"{key}.wav").encode()]
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
