import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import functools


def diff(inputs, axis=-1):

    begin_back = [0] * inputs.shape.rank
    begin_front = [0] * inputs.shape.rank
    begin_front[axis] = 1

    size = inputs.shape.as_list()
    size[axis] -= 1

    back = tf.slice(inputs, begin_back, size)
    front = tf.slice(inputs, begin_front, size)

    return front - back


def unwrap(phases, discont=np.pi, axis=-1):

    diffs = diff(phases, axis=axis)
    mods = tf.mod(diffs + np.pi, 2 * np.pi) - np.pi
    indices = tf.logical_and(tf.equal(mods, -np.pi), tf.greater(diffs, 0))
    mods = tf.where(indices, tf.ones_like(mods) * np.pi, mods)
    corrects = mods - diffs
    cumsums = tf.cumsum(corrects, axis=axis)

    shape = phases.shape.as_list()
    shape[axis] = 1

    cumsums = tf.concat([tf.zeros(shape), cumsums], axis=axis)

    return phases + cumsums


def instantaneous_frequency(phases, axis=-2):

    unwrapped = unwrap(phases, axis=axis)
    diffs = diff(unwrapped, axis=axis)

    begin = [0] * unwrapped.shape.rank

    size = unwrapped.shape.as_list()
    size[axis] = 1

    unwrapped = tf.slice(unwrapped, begin, size)
    diffs = tf.concat([unwrapped, diffs], axis=axis) / np.pi

    return diffs


def convert_to_spectrograms(waveforms, waveform_length, sample_rate, spectrogram_shape, overlap):

    def normalize(inputs, mean, std):
        return (inputs - mean) / std
    # =========================================================================================
    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length
    # =========================================================================================
    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    waveforms = tf.pad(waveforms, [[0, 0], [num_samples - waveform_length, 0]])
    # =========================================================================================
    stfts = tf.signal.stft(
        signals=waveforms,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=functools.partial(
            tf.signal.hann_window,
            periodic=True
        )
    )
    # =========================================================================================
    # discard_dc
    stfts = stfts[..., 1:]
    # =========================================================================================
    magnitude_spectrograms = tf.abs(stfts)
    phase_spectrograms = tf.angle(stfts)
    # =========================================================================================
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_freq_bins,
        num_spectrogram_bins=num_freq_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0,
        upper_edge_hertz=sample_rate / 2
    )
    mel_magnitude_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, axes=1)
    mel_magnitude_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    mel_phase_spectrograms = tf.tensordot(phase_spectrograms, linear_to_mel_weight_matrix, axes=1)
    mel_phase_spectrograms.set_shape(phase_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # =========================================================================================
    log_mel_magnitude_spectrograms = tf.log(mel_magnitude_spectrograms + 1e-6)
    mel_instantaneous_frequencies = instantaneous_frequency(mel_phase_spectrograms)
    # =========================================================================================
    log_mel_magnitude_spectrograms = normalize(log_mel_magnitude_spectrograms, -4, 10)
    mel_instantaneous_frequencies = normalize(mel_instantaneous_frequencies, 0, 1)
    # =========================================================================================
    return log_mel_magnitude_spectrograms, mel_instantaneous_frequencies


def convert_to_waveforms(log_mel_magnitude_spectrograms, mel_instantaneous_frequencies, waveform_length, sample_rate, spectrogram_shape, overlap):

    def unnormalize(inputs, mean, std):
        return inputs * std + mean
    # =========================================================================================
    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length
    # =========================================================================================
    log_mel_magnitude_spectrograms = unnormalize(log_mel_magnitude_spectrograms, -4, 10)
    mel_instantaneous_frequencies = unnormalize(mel_instantaneous_frequencies, 0, 1)
    # =========================================================================================
    mel_magnitude_spectrograms = tf.exp(log_mel_magnitude_spectrograms)
    mel_phase_spectrograms = tf.cumsum(mel_instantaneous_frequencies * np.pi, axis=-2)
    # =========================================================================================
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_freq_bins,
        num_spectrogram_bins=num_freq_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0,
        upper_edge_hertz=sample_rate / 2
    )
    mel_to_linear_weight_matrix = tfp.math.pinv(linear_to_mel_weight_matrix)
    magnitudes = tf.tensordot(mel_magnitude_spectrograms, mel_to_linear_weight_matrix, axes=1)
    magnitudes.set_shape(mel_magnitude_spectrograms.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
    phase_spectrograms = tf.tensordot(mel_phase_spectrograms, mel_to_linear_weight_matrix, axes=1)
    phase_spectrograms.set_shape(mel_phase_spectrograms.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
    # =========================================================================================
    stfts = tf.complex(magnitudes, 0.0) * tf.complex(tf.cos(phase_spectrograms), tf.sin(phase_spectrograms))
    # =========================================================================================
    # discard_dc
    stfts = tf.pad(stfts, [[0, 0], [0, 0], [1, 0]])
    # =========================================================================================
    waveforms = tf.signal.inverse_stft(
        stfts=stfts,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.inverse_stft_window_fn(
            frame_step=frame_step,
            forward_window_fn=functools.partial(
                tf.signal.hann_window,
                periodic=True
            )
        )
    )
    # =========================================================================================
    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    waveforms = waveforms[:, num_samples - waveform_length:]
    # =========================================================================================
    return waveforms
