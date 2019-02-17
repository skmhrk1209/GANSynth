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
    padding_left = padding // 2
    padding_right = padding - padding_left
    wave = tf.pad(wave, [[padding_left, padding_right]])
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


def preprocess(wave, one_hot_label, audio_length, spectrogram_shape, overlap, sampling_rate):
    # =========================================================================================
    audio_length = audio_length
    spectrogram_shape = spectrogram_shape
    overlap = overlap
    sampling_rate = sampling_rate
    # =========================================================================================
    time_steps, freq_bins = spectrogram_shape
    # 2のべき乗はbinary表現でた1を1つだけ持つ
    if not bin(freq_bins).count('1') == 1:
        raise ValueError(
            "Wrong spec_shape. Number of frequency bins must be "
            "a power of 2, not {}".format(freq_bins)
        )
    # 周波数ビンはナイキストレートで折り返される
    # ナイキストレートより先は意味を持たない
    frame_length = freq_bins * 2
    frame_step = int((1. - overlap) * frame_length)
    # =========================================================================================
    # STFTのためのpadding計算
    num_samples = frame_step * (time_steps - 1) + frame_length
    if num_samples < audio_length:
        raise ValueError(
            "Wrong audio length. Number of STFT samples {} should be "
            "greater equal audio lengeth {}".format(num_samples, audio_length)
        )
    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    padding = num_samples - audio_length
    padding_left = padding
    padding_right = padding - padding_left
    # =========================================================================================
    # Convert from waves to complex stfts
    # waves: Tensor of the waveform, shape [batch, time, 1].
    # stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].
    wave_padded = tf.pad(
        tensor=waves,
        paddings=[[padding_left, padding_right]]
    )
    stfts = tf.contrib.signal.stft(
        signals=wave_padded,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length,
        pad_end=False
    )[:, 1:]
    stft_shape = stfts.get_shape().as_list()
    if stft_shape != spectrogram_shape:
        raise ValueError(
            "Spectrogram returned the wrong shape {}, is not the same as the "
            "constructor spec_shape {}.".format(stft_shape, spectrogram_shape)
        )
    # =========================================================================================
    # Converts stfts to specgrams.
    # stfts: Complex64 tensor of stft, shape [batch, time, freq, 1].
    # specgrams: Tensor of log magnitudes and instantaneous frequencies, shape [batch, time, freq, 2].
    log_magnitude_spectrograms = tf.log(tf.abs(stfts) + 1e-6)
    phase_angle = tf.angle(stfts) / np.pi
    spectrograms = tf.concat([
        tf.expand_dims(log_magnitude_spectrograms, axis=-1),
        tf.expand_dims(phase_angle, axis=-1)
    ], axis=-1)


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
    dataset = dataset.map(
        map_func=functools.partial(
            preprocess,
            audio_length=64000,
            spectrogram_shape=(256, 512),
            overlap=0.75,
            sample_rate=16000
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

        wave, one_hot_label = input_fn(
            filenames=["nsynth_test.tfrecord"],
            batch_size=100,
            num_epochs=1,
            shuffle=False
        )

        session.run(tf.tables_initializer())
        print(session.run([wave, one_hot_label]))
