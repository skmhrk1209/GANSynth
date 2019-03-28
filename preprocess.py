import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import skimage.io
import scipy.io.wavfile
import functools
import pathlib
import os
import spectral_ops

tf.logging.set_verbosity(tf.logging.INFO)


def convert_to_spectrograms(waveform_generator, waveform_length, sample_rate, spectrogram_shape, overlap):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    def preprocess(filenames, waveforms):

        def normalize(inputs, mean, std):
            return (inputs - mean) / std
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
        stfts = stfts[:, :, 1:]
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
        mel_instantaneous_frequencies = spectral_ops.instantaneous_frequency(mel_phase_spectrograms)
        # =========================================================================================
        log_mel_magnitude_spectrograms = normalize(log_mel_magnitude_spectrograms, -4, 10)
        mel_instantaneous_frequencies = normalize(mel_instantaneous_frequencies, 0, 1)
        # =========================================================================================
        return filenames, log_mel_magnitude_spectrograms, mel_instantaneous_frequencies

    dataset = tf.data.Dataset.from_generator(
        generator=waveform_generator,
        output_types=(tf.string, tf.float32),
        output_shapes=([], [waveform_length])
    )
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


def main(waveform_dir, magnitude_spectrogram_dir, instantaneous_frequency_dir):

    waveform_dir = pathlib.Path(waveform_dir)
    magnitude_spectrogram_dir = pathlib.Path(magnitude_spectrogram_dir)
    instantaneous_frequency_dir = pathlib.Path(instantaneous_frequency_dir)

    if not magnitude_spectrogram_dir.exists():
        magnitude_spectrogram_dir.mkdir(parents=True, exist_ok=True)

    if not instantaneous_frequency_dir.exists():
        instantaneous_frequency_dir.mkdir(parents=True, exist_ok=True)

    with tf.Graph().as_default():

        def normalize(inputs, mean, std):
            return (inputs - mean) / std

        def unnormalize(inputs, mean, std):
            return inputs * std + mean

        def waveform_generator():
            mean = (np.iinfo(np.int16).max + np.iinfo(np.int16).min) / 2
            std = (np.iinfo(np.int16).max - np.iinfo(np.int16).min) / 2
            for filename in sorted(waveform_dir.glob("*.wav")):
                waveform = scipy.io.wavfile.read(filename)[1]
                waveform = normalize(waveform, mean, std)
                yield (filename.stem, waveform)

        spectrograms = convert_to_spectrograms(
            waveform_generator=waveform_generator,
            waveform_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        )

        with tf.Session() as session:

            while True:
                try:
                    for filename, magnitude_spectrogram, instantaneous_frequency in zip(*session.run(spectrograms)):
                        skimage.io.imsave(
                            fname=magnitude_spectrogram_dir / "{}.jpg".format(filename.decode()),
                            arr=np.clip(unnormalize(magnitude_spectrogram, 0.5, 0.5), 0, 1)
                        )
                        skimage.io.imsave(
                            fname=instantaneous_frequency_dir / "{}.jpg".format(filename.decode()),
                            arr=np.clip(unnormalize(instantaneous_frequency, 0.5, 0.5), 0, 1)
                        )
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":

    main(
        waveform_dir="nsynth-train/audio",
        magnitude_spectrogram_dir="nsynth-train/magnitude_spectrograms",
        instantaneous_frequency_dir="nsynth-train/instantaneous_frequencies"
    )
    main(
        waveform_dir="nsynth-valid/audio",
        magnitude_spectrogram_dir="nsynth-valid/magnitude_spectrograms",
        instantaneous_frequency_dir="nsynth-valid/instantaneous_frequencies"
    )
    main(
        waveform_dir="nsynth-test/audio",
        magnitude_spectrogram_dir="nsynth-test/magnitude_spectrograms",
        instantaneous_frequency_dir="nsynth-test/instantaneous_frequencies"
    )
