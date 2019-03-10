import tensorflow as tf
import os
from utils import Struct


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


def nsynth_input_fn(filenames, batch_size, num_epochs, shuffle, pitches):

    index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitches), dtype=tf.int32)

    def parse_example(example):

        features = Struct(tf.parse_single_example(
            serialized=example,
            features=dict(
                path1=tf.FixedLenFeature([], dtype=tf.string),
                path2=tf.FixedLenFeature([], dtype=tf.string),
                pitch=tf.FixedLenFeature([], dtype=tf.int64),
                source=tf.FixedLenFeature([], dtype=tf.int64)
            )
        ))

        image1 = tf.read_file(features.path1)
        image1 = tf.image.decode_jpeg(image1, channels=1)
        image1 = tf.reshape(image1, [1, 128, 1024])

        image2 = tf.read_file(features.path2)
        image2 = tf.image.decode_jpeg(image2, channels=1)
        image2 = tf.reshape(image2, [1, 128, 1024])

        image = tf.concat([image1, image2], axis=0)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = linear_map(image, 0., 1., -1., 1.)

        label = index_table.lookup(features.pitch)
        label = tf.one_hot(label, len(pitches))

        return image, label, features.pitch, features.source

    dataset = tf.data.TFRecordDataset(filenames)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=sum([
                len(list(tf.io.tf_record_iterator(filename)))
                for filename in filenames
            ]),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.map(
        map_func=parse_example,
        num_parallel_calls=os.cpu_count()
    )
    # filter just acoustic instruments and just pitches 24-84 (as in the paper)
    dataset = dataset.filter(lambda image, label, pitch, source: tf.logical_and(
        x=tf.equal(source, 0),
        y=tf.logical_and(
            x=tf.greater_equal(pitch, min(pitches)),
            y=tf.less_equal(pitch, max(pitches))
        )
    ))
    dataset = dataset.map(
        map_func=lambda image, label, pitch, source: (image, label),
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()

    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, tf.data.experimental.make_saveable_from_iterator(iterator))
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()
