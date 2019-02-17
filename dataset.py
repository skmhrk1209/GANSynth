import tensorflow as tf
import numpy as np
import spectral_ops
import functools
import os


def parse_example(example, audio_length, pitches, index_table):
    # =========================================================================================
    # reference: https://magenta.tensorflow.org/datasets/nsynth
    features = tf.parse_single_example(
        serialized=example,
        features={
            "audio": tf.FixedLenFeature([audio_length], dtype=tf.float32),
            "pitch": tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    # =========================================================================================
    # wave
    wave = features["audio"]
    # force audio length
    padding = tf.maximum(0, audio_length - tf.shape(wave)[0])
    padding_left = padding // 2
    padding_right = padding - padding_left
    wave = tf.pad(wave, [[padding_left, padding_right]])
    wave = wave[:audio_length]
    wave.set_shape([audio_length])
    # =========================================================================================
    # one-hot label
    label = features["pitch"]
    label = index_table.lookup(label)
    label = tf.one_hot(label, len(pitches))

    return wave, label


def preprocess(wave, label, audio_length, spectrogram_shape, overlap, sample_rate, mel_downscale):
    # =========================================================================================
    time_steps, num_freq_bins = spectrogram_shape
    # power of two only has 1 nonzero in binary representation
    if not bin(num_freq_bins).count('1') == 1:
        raise ValueError(
            "Wrong spectrogram_shape. Number of frequency bins must be "
            "a power of 2, not {}".format(num_freq_bins)
        )
    # trim the Nyquist frequency
    frame_length = num_freq_bins * 2
    frame_step = int((1. - overlap) * frame_length)
    # =========================================================================================
    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    num_samples = frame_step * (time_steps - 1) + frame_length
    if num_samples < audio_length:
        raise ValueError(
            "Wrong audio length. Number of STFT samples {} should be "
            "greater equal audio lengeth {}.".format(num_samples, audio_length)
        )
    padding = num_samples - audio_length
    padding_left = padding
    padding_right = padding - padding_left
    # =========================================================================================
    # convert from waves to complex stfts
    # wave: tensor of the waveform, shape [time]
    # stft: complex64 tensor of stft, shape [time, freq]
    wave_padded = tf.pad(
        tensor=wave,
        paddings=[[padding_left, padding_right]]
    )
    stft = tf.contrib.signal.stft(
        signals=wave_padded,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length,
        pad_end=False
    )[:, 1:]
    stft_shape = stft.shape.as_list()
    if stft_shape != spectrogram_shape:
        raise ValueError(
            "Spectrogram returned the wrong shape {}, is not the same as the "
            "constructor spectrogram_shape {}.".format(stft_shape, spectrogram_shape)
        )
    # =========================================================================================
    # converts stft to mel spectrogram
    # stft: complex64 tensor of stft
    # shape [time, freq]
    # mel spectrogram: tensor of log magnitudes and instantaneous frequencies
    # shape [time, freq, 2], mel scaling of frequencies
    magnitude_spectrogram = tf.abs(stft)
    instantaneous_frequency = spectral_ops.instantaneous_frequency(tf.angle(stft))

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_freq_bins // mel_downscale,
        num_spectrogram_bins=num_freq_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate / 2.0
    )
    log_mel_spectrogram = tf.log(tf.tensordot(
        a=magnitude_spectrogram,
        b=linear_to_mel_weight_matrix,
        axes=1
    ) + 1e-6)
    mel_instantaneous_frequency = tf.tensordot(
        a=instantaneous_frequency,
        b=linear_to_mel_weight_matrix,
        axes=1
    )
    log_mel_spectrogram.set_shape(
        magnitude_spectrogram.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]
        )
    )
    print(mel_instantaneous_frequency.shape)
    data = tf.concat([
        tf.expand_dims(log_mel_spectrogram, axis=-1),
        tf.expand_dims(mel_instantaneous_frequency, axis=-1)
    ], axis=-1)

    return data, label


def input_fn(filenames, batch_size, num_epochs, shuffle,
             audio_length, pitches, spectrogram_shape,
             overlap, sample_rate, mel_downscale):

    dataset = tf.data.TFRecordDataset(
        filenames=filenames,
        # num_parallel_reads=os.cpu_count()
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=sum([
                len(list(tf.python_io.tf_record_iterator(filename)))  # kokomo
                for filename in filenames
            ]),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.map(
        map_func=functools.partial(
            parse_example,
            audio_length=audio_length,
            pitches=pitches,
            index_table=tf.contrib.lookup.index_table_from_tensor(
                mapping=sorted(pitches),
                dtype=tf.int64
            )
        ),
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.map(
        map_func=functools.partial(
            preprocess,
            audio_length=audio_length,
            spectrogram_shape=spectrogram_shape,
            overlap=overlap,
            sample_rate=sample_rate,
            mel_downscale=mel_downscale
        ),
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()


if __name__ == "__main__":

    with tf.Session() as session:

        data, label = input_fn(
            filenames=["nsynth_test.tfrecord"],
            batch_size=100,
            num_epochs=1,
            shuffle=False
        )

        session.run(tf.tables_initializer())
        print(session.run([data, label]))
