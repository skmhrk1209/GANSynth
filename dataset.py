import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import os
from utils import Struct
import pathlib


def nsynth_input_fn(directory, waveform_length, pitches, sources, batch_size, num_epochs, shuffle, buffer_size):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length
    index_table = tf.contrib.lookup.index_table_from_tensor(range(*pitch_range), dtype=tf.int32)

    def normalize(inputs, mean, std):
        return (inputs - mean) / std

    def unnormalize(inputs, mean, std):
        return inputs * std + mean

    def generator():
        mean = (np.iinfo(np.int16).max + np.iinfo(np.int16).min) / 2
        std = (np.iinfo(np.int16).max - np.iinfo(np.int16).min) / 2
        for filename in pathlib.Path(directory).glob("*.wav"):
            instrument, pitch, _ = filename.split("-")
            _, source, _ = instrument.split("_")
            if pitch in pitches and source in sources:
                _, waveform = scipy.io.wavfile.read(filename)
                waveform = normalize(waveform, mean, std)
                label = np.squeeze(np.where(np.asanyarray(pitches) == pitch))
                label = np.eye(len(pitches))[label]
                yield waveform, label

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=tf.float32,
        output_shapes=[waveform_length]
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()

    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, tf.data.experimental.make_saveable_from_iterator(iterator))
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()
