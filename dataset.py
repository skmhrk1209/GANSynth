import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import pathlib


def nsynth_input_fn(directory, pitches, sources, batch_size, num_epochs, shuffle, buffer_size):

    def normalize(inputs, mean, std):
        return (inputs - mean) / std

    def unnormalize(inputs, mean, std):
        return inputs * std + mean

    def generator():
        mean = (np.iinfo(np.int16).max + np.iinfo(np.int16).min) / 2
        std = (np.iinfo(np.int16).max - np.iinfo(np.int16).min) / 2
        for filename in pathlib.Path(directory).glob("*.wav"):
            instrument, pitch, _ = str(filename.stem).split("-")
            *_, source, _ = instrument.split("_")
            if int(pitch) in pitches and source in sources:
                _, waveform = scipy.io.wavfile.read(filename)
                waveform = normalize(waveform, mean, std)
                label = np.squeeze(np.where(pitches == pitch))
                label = np.eye(len(pitches))[label]
                yield waveform, label

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=([64000], [len(pitches)])
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
