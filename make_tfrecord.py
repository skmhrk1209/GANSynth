import tensorflow as tf
import sys


def main(input_filename, output_filename):
    with tf.io.TFRecordWriter(output_filename) as writer:
        with open(input_filename) as file:
            for line in file:
                path1, path2, pitch, source = line.split()
                writer.write(record=tf.train.Example(features=tf.train.Features(feature=dict(
                    path1=tf.train.Feature(bytes_list=tf.train.BytesList(value=[path1.encode("utf-8")])),
                    path2=tf.train.Feature(bytes_list=tf.train.BytesList(value=[path2.encode("utf-8")])),
                    pitch=tf.train.Feature(int64_list=tf.train.Int64List(value=[int(pitch)])),
                    source=tf.train.Feature(int64_list=tf.train.Int64List(value=[int(source)]))
                ))).SerializeToString())


if __name__ == "__main__":
    main(*sys.argv[1:])
