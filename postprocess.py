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


def nsynth_output_fn(images, pitches, audio_length, sample_rate, spectrogram_shape, overlap):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    def postprocess(images):
        # =========================================================================================
        log_mel_magnitudes, mel_instantaneous_frequencies = tf.unstack(images, axis=1)
        # =========================================================================================
        log_mel_magnitudes = linear_map(log_mel_magnitudes, -1.0, 1.0, -14.0, 6.0)
        mel_instantaneous_frequencies = linear_map(mel_instantaneous_frequencies, -1.0, 1.0, -1.0, 1.0)
        # =========================================================================================
        mel_magnitudes = tf.exp(log_mel_magnitudes)
        mel_phases = tf.cumsum(mel_instantaneous_frequencies * np.pi, axis=-2)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_freq_bins,
            num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=0.0,
            upper_edge_hertz=sample_rate / 2.0
        )
        mel_to_linear_weight_matrix = tfp.math.pinv(linear_to_mel_weight_matrix)
        magnitudes = tf.tensordot(mel_magnitudes, mel_to_linear_weight_matrix, axes=1)
        magnitudes.set_shape(mel_magnitudes.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
        phases = tf.tensordot(mel_phases, mel_to_linear_weight_matrix, axes=1)
        phases.set_shape(mel_phases.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
        # =========================================================================================
        stfts = tf.complex(magnitudes, 0.) * tf.complex(tf.cos(phases), tf.sin(phases))
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

    dataset = tf.data.Dataset.tensors(images)
    dataset = dataset.map(
        map_func=postprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_hot_iterator()

    return iterator.get_next()


def main(in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with tf.Graph().as_default():

        waveforms = nsynth_output_fn(
            image=list(map(skimage.io.imread, glob.glob(os.path.join(in_dir, "*")))),
            audio_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        )

        with tf.Session() as session:

            for i, waveform in enumerate(session.run(waveforms)):

                scipy.io.wavfile.write(
                    filename="{}.wav".format(i),
                    rate=16000,
                    data=waveform
                )


if __name__ == "__main__":
    main(*sys.argv[1:])
