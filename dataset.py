import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import functools
import pathlib
import json
import os
import spectral_ops
from utils import Struct


def nsynth_input_fn(filenames, pitches, sources, batch_size, num_epochs, shuffle,
                    waveform_length, sample_rate, spectrogram_shape, overlap):

    index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitches), dtype=tf.int32)

    def normalize(inputs, mean, std):
        return (inputs - mean) / std

    def unnormalize(inputs, mean, std):
        return inputs * std + mean

    def parse_example(example):

        features = Struct(tf.parse_single_example(
            serialized=example,
            features=dict(
                path=tf.FixedLenFeature([], dtype=tf.string),
                pitch=tf.FixedLenFeature([], dtype=tf.int64),
                source=tf.FixedLenFeature([], dtype=tf.int64)
            )
        ))

        waveform = tf.read_file(features.path)
        waveform = tf.contrib.ffmpeg.decode_audio(
            contents=waveform,
            file_format="wav",
            samples_per_second=sample_rate,
            channel_count=1
        )
        waveform = tf.squeeze(waveform)
        waveform.set_shape([waveform_length])

        label = index_table.lookup(features.pitch)
        label = tf.one_hot(label, len(pitches))

        return waveform, label, features.pitch, features.source

    def preprocess(waveforms, labels):

        magnitude_spectrograms, instantaneous_frequencies = spectral_ops.convert_to_spactrograms(
            waveforms=waveforms,
            waveform_length=waveform_length,
            sample_rate=sample_rate,
            spectrogram_shape=spectrogram_shape,
            overlap=overlap
        )

        return waveforms, magnitude_spectrograms, instantaneous_frequencies, labels

    dataset = tf.data.TFRecordDataset(filenames=filenames)
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
    dataset = dataset.filter(
        predicate=lambda waveform, label, pitch, source: tf.logical_and(
            x=tf.equal(source, 0),
            y=tf.logical_and(
                x=tf.greater_equal(pitch, min(pitches)),
                y=tf.less_equal(pitch, max(pitches))
            )
        )
    )
    dataset = dataset.map(
        map_func=lambda waveform, label, pitch, source: (waveform, label),
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.batch(
        batch_size=batch_size,
        drop_remainder=True
    )
    dataset = dataset.map(
        map_func=preprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()

    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, tf.data.experimental.make_saveable_from_iterator(iterator))
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()
