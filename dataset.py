import tensorflow as tf
import numpy as np
import spectral_ops
import functools
import os


class NSynth(object):

    def __init__(self, pitch_counts, audio_length, sample_rate, spectrogram_shape, overlap, mel_downscale):

        self.pitch_counts = pitch_counts
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.spectrogram_shape = spectrogram_shape
        self.overlap = overlap
        self.mel_downscale = mel_downscale
        self.index_table = tf.contrib.lookup.index_table_from_tensor(
            mapping=sorted(pitch_counts),
            dtype=tf.int32
        )

    def parse_example(self, example):
        # =========================================================================================
        # reference: https://magenta.tensorflow.org/datasets/nsynth
        features = tf.parse_single_example(
            serialized=example,
            features={
                "audio": tf.FixedLenFeature([self.audio_length], dtype=tf.float32),
                "pitch": tf.FixedLenFeature([], dtype=tf.int64),
                "instrument_source": tf.FixedLenFeature([], dtype=tf.int64)
            }
        )
        # =========================================================================================
        # wave
        wave = features["audio"]
        # =========================================================================================
        # pitch
        pitch = features["pitch"]
        # =========================================================================================
        # label
        label = self.index_table.lookup(pitch)
        label = tf.one_hot(label, len(self.pitch_counts))
        # =========================================================================================
        # source
        source = features["instrument_source"]

        return wave, label, pitch, source

    def preprocess(self, waves, labels, pitches, sources):
        # =========================================================================================
        time_steps, num_freq_bins = self.spectrogram_shape
        # power of two only has 1 nonzero in binary representation
        if not bin(num_freq_bins).count("1") == 1:
            raise ValueError(
                "Wrong spectrogram_shape. Number of frequency bins must be "
                "a power of 2, not {}".format(num_freq_bins)
            )
        # trim the Nyquist frequency
        frame_length = num_freq_bins * 2
        frame_step = int((1. - self.overlap) * frame_length)
        # =========================================================================================
        # For Nsynth dataset, we are putting all padding in the front
        # This causes edge effects in the tail
        num_samples = frame_step * (time_steps - 1) + frame_length
        if num_samples < self.audio_length:
            raise ValueError(
                "Wrong audio length. Number of STFT samples {} should be "
                "greater equal audio lengeth {}.".format(num_samples, self.audio_length)
            )
        padding = num_samples - self.audio_length
        padding_left = padding
        padding_right = padding - padding_left
        # =========================================================================================
        # convert from waves to complex stfts
        # waves: tensor of the waveform
        # shape [batch, time]
        # stfts: complex64 tensor of stft
        # shape [batch, time, freq]
        waves = tf.pad(
            tensor=waves,
            paddings=[[0, 0], [padding_left, padding_right]]
        )
        stfts = tf.contrib.signal.stft(
            signals=waves,
            frame_length=frame_length,
            frame_step=frame_step
        )[:, :, 1:]
        stft_shape = stfts.shape.as_list()[1:]
        if stft_shape != self.spectrogram_shape:
            raise ValueError(
                "Spectrogram returned the wrong shape {}, is not the same as the "
                "constructor spectrogram_shape {}.".format(stft_shape, self.spectrogram_shape)
            )
        # =========================================================================================
        # converts stft to mel spectrogram
        # stfts: complex64 tensor of stft
        # shape [batch, time, freq]
        # mel spectrograms: tensor of log magnitudes and instantaneous frequencies
        # shape [batch, time, freq, 2], mel scaling of frequencies
        magnitudes = tf.abs(stfts)
        phases = tf.angle(stfts)

        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_freq_bins // self.mel_downscale,
            num_spectrogram_bins=num_freq_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_magnitudes = tf.tensordot(
            a=magnitudes,
            b=linear_to_mel_weight_matrix,
            axes=1
        )
        mel_magnitudes.set_shape(
            magnitudes.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )
        mel_phases = tf.tensordot(
            a=phases,
            b=linear_to_mel_weight_matrix,
            axes=1
        )
        mel_phases.set_shape(
            phases.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )

        log_mel_magnitudes = tf.log(mel_magnitudes + 1e-6)
        mel_instantaneous_frequencies = spectral_ops.instantaneous_frequency(mel_phases)

        def scale(input, input_min, input_max, output_min, output_max):
            return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)

        log_mel_magnitudes = scale(log_mel_magnitudes, -14.0, 6.0, -1.0, 1.0)

        data = tf.stack([
            log_mel_magnitudes,
            mel_instantaneous_frequencies
        ], axis=1)

        return data, labels

    def real_input_fn(self, filenames, batch_size, num_epochs, shuffle):

        dataset = tf.data.TFRecordDataset(filenames)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=sum([
                    len(list(tf.python_io.tf_record_iterator(filename)))
                    for filename in filenames
                ]),
                reshuffle_each_iteration=True
            )
        dataset = dataset.repeat(count=num_epochs)
        dataset = dataset.map(
            map_func=self.parse_example,
            num_parallel_calls=os.cpu_count()
        )
        # filter just acoustic instruments (as in the paper) and just pitches 24-84
        dataset = dataset.filter(lambda wave, label, pitch, source: tf.logical_and(
            x=tf.equal(source, 0),
            y=tf.logical_and(
                x=tf.greater_equal(pitch, min(self.pitch_counts)),
                y=tf.less_equal(pitch, max(self.pitch_counts))
            )
        ))
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.map(
            map_func=self.preprocess,
            num_parallel_calls=os.cpu_count()
        )
        dataset = dataset.prefetch(buffer_size=1)

        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        return iterator.get_next()

    def fake_input_fn(self, latent_size, batch_size):

        latents = tf.random_normal([batch_size, latent_size])

        labels = tf.one_hot(tf.reshape(tf.multinomial(
            logits=tf.log([tf.cast(list(zip(*sorted(self.pitch_counts.items())))[1], tf.float32)]),
            num_samples=batch_size
        ), [batch_size]), len(self.pitch_counts))

        return latents, labels
