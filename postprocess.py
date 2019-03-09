import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import skimage
import spectral_ops
import functools
import itertools
import pickle
import sys
import os

tf.logging.set_verbosity(tf.logging.INFO)


def linear_map(inputs, in_min, in_max, out_min, out_max):
    return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)


def nsynth_input_fn(filenames, batch_size, num_epochs, shuffle, pitches,
                    audio_length, sample_rate, spectrogram_shape, overlap):

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    def parse_example(example):
        # =========================================================================================
        # reference: https://magenta.tensorflow.org/datasets/nsynth
        features = tf.parse_single_example(
            serialized=example,
            features=dict(
                audio=tf.FixedLenFeature([audio_length], dtype=tf.float32),
                pitch=tf.FixedLenFeature([], dtype=tf.int64),
                instrument_source=tf.FixedLenFeature([], dtype=tf.int64)
            )
        )
        # =========================================================================================
        # waveform
        waveform = features["audio"]
        # =========================================================================================
        # pitch
        pitch = features["pitch"]
        # =========================================================================================
        # source
        source = features["instrument_source"]
        # =========================================================================================

        return waveform, pitch, source

    def preprocess(waveforms, pitches, sources):
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
        magnitudes = tf.abs(stfts)
        phases = tf.angle(stfts)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_freq_bins,
            num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=0.,
            upper_edge_hertz=sample_rate / 2.
        )
        mel_magnitudes = tf.tensordot(magnitudes, linear_to_mel_weight_matrix, axes=1)
        mel_magnitudes.set_shape(magnitudes.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        mel_phases = tf.tensordot(phases, linear_to_mel_weight_matrix, axes=1)
        mel_phases.set_shape(phases.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        # =========================================================================================
        log_mel_magnitudes = tf.log(mel_magnitudes + 1e-6)
        mel_instantaneous_frequencies = spectral_ops.instantaneous_frequency(mel_phases)
        # =========================================================================================
        log_mel_magnitudes = linear_map(log_mel_magnitudes, -14., 6., -1., 1.)
        mel_instantaneous_frequencies = linear_map(mel_instantaneous_frequencies, -1., 1., -1., 1.)
        # =========================================================================================
        images = tf.stack([log_mel_magnitudes, mel_instantaneous_frequencies], axis=1)
        # =========================================================================================

        return images, pitches

    def postprocess(images):
        # =========================================================================================
        log_mel_magnitudes, mel_instantaneous_frequencies = tf.unstack(images, axis=1)
        # =========================================================================================
        log_mel_magnitudes = linear_map(log_mel_magnitudes, -1., 1., -14., 6.)
        mel_instantaneous_frequencies = linear_map(mel_instantaneous_frequencies, -1., 1., -1., 1.)
        # =========================================================================================
        mel_magnitudes = tf.exp(log_mel_magnitudes)
        mel_phases = tf.cumsum(mel_instantaneous_frequencies * np.pi, axis=-2)
        # =========================================================================================
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_freq_bins,
            num_spectrogram_bins=num_freq_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=0.,
            upper_edge_hertz=sample_rate / 2.
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
        map_func=parse_example,
        num_parallel_calls=os.cpu_count()
    )
    # filter just acoustic instruments and just pitches 24-84 (as in the paper)
    dataset = dataset.filter(lambda waveform, pitch, source: tf.logical_and(
        x=tf.equal(source, 0),
        y=tf.logical_and(
            x=tf.greater_equal(pitch, min(pitches)),
            y=tf.less_equal(pitch, max(pitches))
        )
    ))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(
        map_func=preprocess,
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()


def main(filename, directory):

    directory1 = os.path.join(directory, "log_mel_magnitudes")
    directory2 = os.path.join(directory, "mel_instantaneous_frequencies")

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    if not os.path.exists(directory2):
        os.makedirs(directory2)

    with tf.Graph().as_default():

        tf.set_random_seed(0)

        with open("pitch_counts.pickle", "rb") as file:
            pitch_counts = pickle.load(file)

        images, pitches = nsynth_input_fn(
            filenames=[filename],
            batch_size=128,
            num_epochs=1,
            shuffle=False,
            pitches=pitch_counts.keys(),
            audio_length=64000,
            sample_rate=16000,
            spectrogram_shape=[128, 1024],
            overlap=0.75
        )

        with tf.Session() as session:

            session.run(tf.tables_initializer())

            with open(os.path.join(directory, "gt.txt"), "w") as file:

                try:
                    tf.logging.info("preprocessing started")
                    i = 0
                    while True:
                        for image, pitch in zip(*session.run([images, pitches])):
                            path1 = os.path.join(directory1, "{}.jpg".format(i))
                            path2 = os.path.join(directory2, "{}.jpg".format(i))
                            image1 = linear_map(image[0], -1., 1., 0., 255.).clip(0., 255.).astype(np.uint8)
                            image2 = linear_map(image[1], -1., 1., 0., 255.).clip(0., 255.).astype(np.uint8)
                            skimage.io.imsave(path1, image1)
                            skimage.io.imsave(path2, image2)
                            file.write("{} {} {}\n".format(path1, path2, pitch))
                            i += 1

                except tf.errors.OutOfRangeError:
                    tf.logging.info("preprocessing completed")


if __name__ == "__main__":
    main(*sys.argv[1:])
