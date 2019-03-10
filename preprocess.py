import tensorflow as tf
import numpy as np
import skimage
import glob
import sys
import os
import spectral_ops
import scipy.io.wavfile

tf.logging.set_verbosity(tf.logging.INFO)


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


def convert_to_spectrograms(waveforms, audio_length, sample_rate, spectrogram_shape, overlap):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    def preprocess(waveforms):
        # =========================================================================================
        # For Nsynth dataset, we are putting all padding in the front
        # This causes edge effects in the tail
        waveforms = tf.pad(waveforms, [[0, 0], [num_samples - audio_length, 0]])
        # =========================================================================================
        stfts = tf.contrib.signal.stft(
            signals=waveforms,
            frame_length=frame_length,
            frame_step=frame_step
        )
        # =========================================================================================
        # discard_dc
        stfts = stfts[:, :, 1:]
        # =========================================================================================
        magnitude_spectrograms = tf.abs(stfts)
        phase_spectrograms = tf.angle(stfts)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_freq_bins,
            num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=sample_rate / 2.0
        )
        mel_magnitude_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, axes=1)
        mel_magnitude_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        mel_phase_spectrograms = tf.tensordot(phase_spectrograms, linear_to_mel_weight_matrix, axes=1)
        mel_phase_spectrograms.set_shape(phase_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # =========================================================================================
        log_mel_magnitude_spectrograms = tf.log(mel_magnitude_spectrograms + 1e-6)
        mel_instantaneous_frequencies = spectral_ops.instantaneous_frequency(mel_phase_spectrograms)
        # =========================================================================================
        log_mel_magnitude_spectrograms = linear_map(log_mel_magnitude_spectrograms, -14.0, 6.0, 0.0, 1.0)
        mel_instantaneous_frequencies = linear_map(mel_instantaneous_frequencies, -1.0, 1.0, 0.0, 1.0)
        # =========================================================================================
        return log_mel_magnitude_spectrograms, mel_instantaneous_frequencies

    dataset = tf.data.Dataset.from_tensor_slices(waveforms)
    dataset = dataset.batch(
        batch_size=100,
        drop_remainder=False
    )
    dataset = dataset.map(
        map_func=preprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def main(waveform_dir, log_mel_magnitude_spectrogram_dir, mel_instantaneous_frequency_dir):

    if not os.path.exists(log_mel_magnitude_spectrogram_dir):
        os.makedirs(log_mel_magnitude_spectrogram_dir)

    if not os.path.exists(mel_instantaneous_frequency_dir):
        os.makedirs(mel_instantaneous_frequency_dir)

    with tf.Graph().as_default():

        filenames = sorted(glob.glob(os.path.join(waveform_dir, "*")))
        waveforms = np.array([scipy.io.wavfile.read(filename)[1] for filename in filenames])
        waveforms = linear_map(waveforms.astype(np.float32), np.iinfo(np.int16).min, np.iinfo(np.int16).max, -1.0, 1.0)

        log_mel_magnitude_spectrograms, mel_instantaneous_frequencies = convert_to_spectrograms(
            waveforms=waveforms,
            audio_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        )

        with tf.Session() as session:

            try:
                tf.logging.info("preprocessing started")

                while True:
                    for filename, (log_mel_magnitude_spectrogram, mel_instantaneous_frequency) in zip(
                        filenames, zip(*session.run([log_mel_magnitude_spectrograms, mel_instantaneous_frequencies]))
                    ):
                        skimage.io.imsave(os.path.join(
                            log_mel_magnitude_spectrogram_dir,
                            "{}.jpg".format(os.path.splitext(os.path.basename(filename))[0])
                        ), log_mel_magnitude_spectrogram.clip(0.0, 1.0))

                        skimage.io.imsave(os.path.join(
                            mel_instantaneous_frequency_dir,
                            "{}.jpg".format(os.path.splitext(os.path.basename(filename))[0])
                        ), mel_instantaneous_frequency.clip(0.0, 1.0))

            except tf.errors.OutOfRangeError:
                tf.logging.info("preprocessing completed")


if __name__ == "__main__":
    main(*sys.argv[1:])
