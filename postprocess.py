import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import glob
import sys
import os
import skimage
import scipy.io.wavfile
import spectral_ops

tf.logging.set_verbosity(tf.logging.INFO)


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


def convert_to_waveform(log_mel_magnitude_spectrograms, mel_instantaneous_frequencies,
                        audio_length, sample_rate, spectrogram_shape, overlap):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    def postprocess(log_mel_magnitude_spectrograms, mel_instantaneous_frequencies):
        # =========================================================================================
        log_mel_magnitude_spectrograms = linear_map(log_mel_magnitude_spectrograms, 0.0, 1.0, -14.0, 6.0)
        mel_instantaneous_frequencies = linear_map(mel_instantaneous_frequencies, 0.0, 1.0, -1.0, 1.0)
        # =========================================================================================
        mel_magnitude_spectrograms = tf.exp(log_mel_magnitude_spectrograms)
        mel_phase_spectrograms = tf.cumsum(mel_instantaneous_frequencies * np.pi, axis=-2)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_freq_bins,
            num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=sample_rate / 2.0
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
        waveforms = tf.contrib.signal.inverse_stft(
            stfts=stfts,
            frame_length=frame_length,
            frame_step=frame_step,
            window_fn=tf.contrib.signal.inverse_stft_window_fn(
                frame_step=frame_step
            )
        )
        # =========================================================================================
        # For Nsynth dataset, we are putting all padding in the front
        # This causes edge effects in the tail
        waveforms = waveforms[:, num_samples - audio_length:]
        # =========================================================================================
        return waveforms

    dataset = tf.data.Dataset.from_tensor_slices((log_mel_magnitude_spectrograms, mel_instantaneous_frequencies))
    dataset = dataset.batch(
        batch_size=100,
        drop_remainder=False
    )
    dataset = dataset.map(
        map_func=postprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def main(log_mel_magnitude_spectrogram_dir, mel_instantaneous_frequency_dir, waveform_dir):

    if not os.path.exists(waveform_dir):
        os.makedirs(waveform_dir)

    with tf.Graph().as_default():

        filenames = sorted(glob.glob(os.path.join(log_mel_magnitude_spectrogram_dir, "*")))
        log_mel_magnitude_spectrograms = np.array([np.squeeze(skimage.io.imread(filename)) for filename in filenames])
        log_mel_magnitude_spectrograms = linear_map(log_mel_magnitude_spectrograms.astype(np.float32), 0.0, 255.0, 0.0, 1.0)

        filenames = sorted(glob.glob(os.path.join(mel_instantaneous_frequency_dir, "*")))
        mel_instantaneous_frequencies = np.array([np.squeeze(skimage.io.imread(filename)) for filename in filenames])
        mel_instantaneous_frequencies = linear_map(mel_instantaneous_frequencies.astype(np.float32), 0.0, 255.0, 0.0, 1.0)

        waveforms = convert_to_waveform(
            log_mel_magnitude_spectrograms=log_mel_magnitude_spectrograms,
            mel_instantaneous_frequencies=mel_instantaneous_frequencies,
            audio_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        )

        with tf.Session() as session:

            try:
                tf.logging.info("preprocessing started")

                while True:
                    for filename, waveform in zip(filenames, session.run(waveforms)):
                        scipy.io.wavfile.write(os.path.join(
                            waveform_dir,
                            "{}.wav".format(os.path.splitext(os.path.basename(filename))[0])
                        ), 16000, waveform)

            except tf.errors.OutOfRangeError:
                tf.logging.info("preprocessing completed")


if __name__ == "__main__":
    main(*sys.argv[1:])
