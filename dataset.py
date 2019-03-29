import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import functools
import pathlib
import json
import os
import spectral_ops


def nsynth_input_fn(directory, pitches, sources, batch_size, num_epochs, shuffle, buffer_size,
                    waveform_length, sample_rate, spectrogram_shape, overlap):

    index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitches), dtype=tf.int32)

    def normalize(inputs, mean, std):
        return (inputs - mean) / std

    def unnormalize(inputs, mean, std):
        return inputs * std + mean

    def generator(directory):
        with open("{}/examples.json".format(directory)) as file:
            examples = json.load(file)
        for example in examples.values():
            if example["pitch"] in pitches and example["instrument_source"] in sources:
                yield "{}/{}.wav".format(directory, example["note_str"]), example["pitch"]

    def decode_audio(filename, pitch):

        waveform = tf.contrib.ffmpeg.decode_audio(
            contents=tf.read_file(filename),
            file_format="wav",
            samples_per_second=sample_rate,
            channel_count=1
        )
        waveform = tf.squeeze(waveform)
        waveform.set_shape([waveform_length])

        label = index_table.lookup(pitch)
        label = tf.one_hot(label, len(pitches))

        return waveform, label

    def convert_to_spactrograms(waveforms, labels):

        magnitude_spectrograms, instantaneous_frequencies = spectral_ops.convert_to_spactrograms(
            waveforms=waveforms,
            waveform_length=waveform_length,
            sample_rate=sample_rate,
            spectrogram_shape=spectrogram_shape,
            overlap=overlap
        )

        return waveforms, magnitude_spectrograms, instantaneous_frequencies, labels

    dataset = tf.data.Dataset.from_generator(
        generator=functools.partial(generator, directory),
        output_types=(tf.string, tf.int64),
        output_shapes=([], [])
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.map(decode_audio, num_parallel_calls=os.cpu_count())
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(convert_to_spactrograms, num_parallel_calls=os.cpu_count())
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()
