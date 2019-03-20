import tensorflow as tf
import os
from utils import Struct


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


def nsynth_real_input_fn(filenames, batch_size, num_epochs, shuffle, pitch_counts):

    index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitch_counts), dtype=tf.int32)

    def parse_example(example):

        features = Struct(tf.parse_single_example(
            serialized=example,
            features=dict(
                path_to_magnitude_spectrogram=tf.FixedLenFeature([], dtype=tf.string),
                path_to_instantaneous_frequency=tf.FixedLenFeature([], dtype=tf.string),
                instrument_source=tf.FixedLenFeature([], dtype=tf.int64),
                pitch=tf.FixedLenFeature([], dtype=tf.int64)
            )
        ))

        magnitude_spectrogram = tf.read_file(features.path_to_magnitude_spectrogram)
        magnitude_spectrogram = tf.image.decode_jpeg(magnitude_spectrogram, channels=1)
        magnitude_spectrogram = tf.reshape(magnitude_spectrogram, [1, 128, 1024])

        instantaneous_frequency = tf.read_file(features.path_to_instantaneous_frequency)
        instantaneous_frequency = tf.image.decode_jpeg(instantaneous_frequency, channels=1)
        instantaneous_frequency = tf.reshape(instantaneous_frequency, [1, 128, 1024])

        image = tf.concat([magnitude_spectrogram, instantaneous_frequency], axis=0)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = linear_map(image, 0.0, 1.0, -1.0, 1.0)

        label = index_table.lookup(features.pitch)
        label = tf.one_hot(label, len(pitch_counts))

        return image, label, features.instrument_source, features.pitch

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
    dataset = dataset.filter(lambda image, label, instrument_source, pitch: tf.logical_and(
        x=tf.equal(instrument_source, 0),
        y=tf.logical_and(
            x=tf.greater_equal(pitch, min(pitch_counts)),
            y=tf.less_equal(pitch, max(pitch_counts))
        )
    ))
    dataset = dataset.map(
        map_func=lambda image, label, instrument_source, pitch: (image, label),
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()

    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, tf.data.experimental.make_saveable_from_iterator(iterator))
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()


def nsynth_fake_input_fn(latent_size, batch_size, pitch_counts):

    index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitch_counts), dtype=tf.int32)

    latents = tf.random_normal([batch_size, latent_size])

    labels = index_table.lookup(tf.reshape(tf.multinomial(
        logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
        num_samples=batch_size
    ), [batch_size]))
    labels = tf.one_hot(labels, len(pitch_counts))

    return latents, labels
