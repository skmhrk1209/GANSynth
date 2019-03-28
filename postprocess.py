import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import skimage
import functools
import pathlib
import os
import scipy.io.wavfile
import spectral_ops

tf.logging.set_verbosity(tf.logging.INFO)


def convert_to_waveform(spectrogram_generator, waveform_length, sample_rate, spectrogram_shape, overlap):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    def postprocess(filename, log_mel_magnitude_spectrograms, mel_instantaneous_frequencies):

        def unnormalize(inputs, mean, std):
            return inputs * std + mean
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
        stfts = tf.complex(magnitudes, 0) * tf.complex(tf.cos(phase_spectrograms), tf.sin(phase_spectrograms))
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
        return filename, waveforms

    dataset = tf.data.Dataset.from_generator(
        generator=spectrogram_generator,
        output_types=(tf.string, tf.float32, tf.float32),
        output_shapes=([], spectrogram_shape, spectrogram_shape)
    )
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


def main(magnitude_spectrogram_dir, instantaneous_frequency_dir, waveform_dir):

    magnitude_spectrogram_dir = Path(magnitude_spectrogram_dir)
    instantaneous_frequency_dir = Path(instantaneous_frequency_dir)
    waveform_dir = Path(waveform_dir)

    if not waveform_dir.exists():
        waveform_dir.mkdir(parents=True, exist_ok=True)

    with tf.Graph().as_default():

        def normalize(inputs, mean, std):
            return (inputs - mean) / std

        def unnormalize(inputs, mean, std):
            return inputs * std + mean

        def spectrogram_generator():
            mean = (np.iinfo(np.uint8).max + np.iinfo(np.uint8).min) / 2
            std = (np.iinfo(np.uint8).max - np.iinfo(np.uint8).min) / 2
            for filename1, filename2 in zip(sorted(magnitude_spectrogram_dir.glob("*.jpg")), sorted(instantaneous_frequency_dir.glob("*.jpg"))):
                assert filename1.name == filename2.name
                magnitude_spectrogram = np.squeeze(skimage.io.imread(filename1))
                magnitude_spectrogram = normalize(magnitude_spectrogram, mean, std)
                instantaneous_frequency = np.squeeze(skimage.io.imread(filename2))
                instantaneous_frequency = normalize(instantaneous_frequency, mean, std)
                yield ((filename1 or filename2).stem, magnitude_spectrogram, instantaneous_frequency)

        waveforms = convert_to_waveform(
            spectrogram_generator=spectrogram_generator,
            waveform_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        )

        with tf.Session() as session:

            while True:
                try:
                    for filename, waveform in zip(*session.run(waveforms)):
                        scipy.io.wavfile.write(
                            filename=waveform_dir / "{}.wav".format(filename.decode()),
                            rate=16000,
                            data=waveform
                        )
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":

    main(
        waveform_dir="samples/waveforms",
        magnitude_spectrogram_dir="samples/magnitude_spectrograms",
        instantaneous_frequency_dir="samples/instantaneous_frequencies"
    )
