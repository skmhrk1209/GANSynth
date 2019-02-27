import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import spectral_ops
import functools
import os


def scale(inputs, in_min, in_max, out_min, out_max):
    inputs = out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)
    return inputs


class NSynth(object):

    def __init__(self, pitch_counts, audio_length, sample_rate, spectrogram_shape, overlap):

        self.pitch_counts = pitch_counts
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.spectrogram_shape = spectrogram_shape
        self.overlap = overlap
        self.index_table = tf.contrib.lookup.index_table_from_tensor(sorted(pitch_counts), dtype=tf.int32)

        self.time_steps, self.num_freq_bins = self.spectrogram_shape
        self.frame_length = self.num_freq_bins * 2
        self.frame_step = int((1 - self.overlap) * self.frame_length)
        self.num_samples = self.frame_step * (self.time_steps - 1) + self.frame_length

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
        # waveform
        waveform = features["audio"]
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
        # =========================================================================================

        return waveform, label, pitch, source

    def preprocess(self, waveforms, labels, pitches, sources):
        # =========================================================================================
        # For Nsynth dataset, we are putting all padding in the front
        # This causes edge effects in the tail
        waveforms = tf.pad(waveforms, [[0, 0], [self.num_samples - self.audio_length, 0]])
        # =========================================================================================
        stfts = tf.contrib.signal.stft(
            signals=waveforms,
            frame_length=self.frame_length,
            frame_step=self.frame_step
        )
        # =========================================================================================
        # discard_dc
        stfts = stfts[:, :, 1:]
        # =========================================================================================
        magnitudes = tf.abs(stfts)
        phases = tf.angle(stfts)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_freq_bins,
            num_spectrogram_bins=self.num_freq_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_magnitudes = tf.tensordot(magnitudes, linear_to_mel_weight_matrix, axes=1)
        mel_magnitudes.set_shape(magnitudes.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        mel_phases = tf.tensordot(phases, linear_to_mel_weight_matrix, axes=1)
        mel_phases.set_shape(phases.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # =========================================================================================
        log_mel_magnitudes = tf.log(mel_magnitudes + 1e-6)
        mel_instantaneous_frequencies = spectral_ops.instantaneous_frequency(mel_phases)
        # =========================================================================================
        log_mel_magnitudes = scale(log_mel_magnitudes, -14.0, 6.0, -1.0, 1.0)
        # =========================================================================================
        images = tf.stack([log_mel_magnitudes, mel_instantaneous_frequencies], axis=1)
        # =========================================================================================

        return images, labels

    def postprocess(self, images):
        # =========================================================================================
        log_mel_magnitudes, mel_instantaneous_frequencies = tf.unstack(images, axis=1)
        # =========================================================================================
        log_mel_magnitudes = scale(log_mel_magnitudes, -1.0, 1.0, -14.0, 6.0)
        # =========================================================================================
        mel_magnitudes = tf.exp(log_mel_magnitudes)
        mel_phases = tf.cumsum(mel_instantaneous_frequencies * np.pi, axis=-2)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_freq_bins,
            num_spectrogram_bins=self.num_freq_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=self.sample_rate / 2.0
        )
        mel_to_linear_weight_matrix = tfp.math.pinv(linear_to_mel_weight_matrix)
        magnitudes = tf.tensordot(mel_magnitudes, mel_to_linear_weight_matrix, axes=1)
        magnitudes.set_shape(mel_magnitudes.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
        phases = tf.tensordot(mel_phases, mel_to_linear_weight_matrix, axes=1)
        phases.set_shape(mel_phases.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
        # =========================================================================================
        stfts = tf.complex(magnitudes, 0.0) * tf.complex(tf.cos(phases), tf.sin(phases))
        # =========================================================================================
        # discard_dc
        stfts = tf.pad(stfts, [[0, 0], [0, 0], [1, 0]])
        # =========================================================================================
        waveforms = tf.contrib.signal.inverse_stft(
            stfts=stfts,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            window_fn=tf.contrib.signal.inverse_stft_window_fn(
                frame_step=self.frame_step
            )
        )
        # =========================================================================================
        # For Nsynth dataset, we are putting all padding in the front
        # This causes edge effects in the tail
        waveforms = waveforms[:, self.num_samples - self.audio_length:]
        # =========================================================================================

        return waveforms

    def real_input_fn(self, filenames, batch_size, num_epochs, shuffle):

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

        labels = tf.one_hot(tf.reshape(tf.random.multinomial(
            logits=tf.log([tf.cast(list(zip(*sorted(self.pitch_counts.items())))[1], tf.float32)]),
            num_samples=batch_size
        ), [batch_size]), len(self.pitch_counts))

        return latents, labels
