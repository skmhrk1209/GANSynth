import tensorflow as tf
import numpy as np
import functools
import os
import spectral_ops
import pitch


def parse_example(example, length, channels, pitches, index_table):
    # =========================================================================================
    # reference: https://magenta.tensorflow.org/datasets/nsynth
    features = tf.parse_single_example(
        serialized=example,
        features={
            "pitch": tf.FixedLenFeature([], dtype=tf.int64),
            "audio": tf.FixedLenFeature([length], dtype=tf.float32),
            "instrument_source": tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    # =========================================================================================
    # wave
    wave = features["audio"]
    # force audio length
    padding = tf.maximum(0, length - tf.shape(wave)[0])
    left_padding = padding // 2
    right_padding = padding - left_padding
    wave = tf.pad(wave, [[left_padding, right_padding]])
    wave = wave[:length]
    # force number of channels
    wave = tf.expand_dims(wave, axis=-1)
    wave = tf.tile(wave, [1, channels])
    # =========================================================================================
    # label
    label = features["pitch"]
    # =========================================================================================
    # one-hot label
    one_hot_label = tf.one_hot(
        indices=index_table.lookup(label),
        depth=len(pitches)
    )
    # =========================================================================================
    # instrument_source
    source = features["instrument_source"]

    return wave, one_hot_label, label, source


def input_fn(filenames, batch_size, num_epochs, shuffle,
             length=64000, channels=1, pitches=range(24, 85)):

    dataset = tf.data.TFRecordDataset(
        filenames=filenames,
        num_parallel_reads=os.cpu_count()
    )
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
        map_func=functools.partial(
            parse_example,
            length=length,
            channels=channels,
            pitches=pitches,
            index_table=tf.contrib.lookup.index_table_from_tensor(
                mapping=sorted(pitches),
                dtype=tf.int64
            )
        ),
        num_parallel_calls=os.cpu_count()
    )
    # Filter just acoustic instruments (as in the paper)
    dataset = dataset.filter(lambda *args: tf.equal(args[-1], 0))
    # Filter just pitches 24-84
    dataset = dataset.filter(lambda *args: tf.greater_equal(args[-2], min(pitches)))
    dataset = dataset.filter(lambda *args: tf.less_equal(args[-2], max(pitches)))
    dataset = dataset.map(lambda *args: args[:-2])

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()


if __name__ == "__main__":

    with tf.Session() as session:

        wave, one_hot_label = input_fn(
            filenames=["nsynth_test.tfrecord"],
            batch_size=100,
            num_epochs=1,
            shuffle=False
        )

        session.run(tf.tables_initializer())
        print(session.run([wave, one_hot_label]))
