import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import functools
import metrics
import glob

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def mel_to_hertz(mel_values):
  """Converts frequencies in `mel_values` from the mel scale to linear scale."""
  return _MEL_BREAK_FREQUENCY_HERTZ * (
      np.exp(np.array(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0)


def hertz_to_mel(frequencies_hertz):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (np.array(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))

def _linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=16000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
    """Returns a matrix to warp linear scale spectrograms to the mel scale.
    Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
    band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
    we compute the matrix at float64 precision and then cast to `dtype`
    at the end. This function can be constant folded by graph optimization
    since there are no Tensor inputs.
    Args:
      num_mel_bins: Int, number of output frequency dimensions.
      num_spectrogram_bins: Int, number of input frequency dimensions.
      sample_rate: Int, sample rate of the audio.
      lower_edge_hertz: Float, lowest frequency to consider.
      upper_edge_hertz: Float, highest frequency to consider.
    Returns:
      Numpy float32 matrix of shape [num_spectrogram_bins, num_mel_bins].
    Raises:
      ValueError: Input argument in the wrong range.
    """
    # Validate input arguments
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
    if num_spectrogram_bins <= 0:
        raise ValueError(
            'num_spectrogram_bins must be positive. Got: %s' % num_spectrogram_bins)
    if sample_rate <= 0.0:
        raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                         (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                         'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                         % (upper_edge_hertz, sample_rate))

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:, np.newaxis]
    # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2)

    lower_edge_mel = band_edges_mel[0:-2]
    center_mel = band_edges_mel[1:-1]
    upper_edge_mel = band_edges_mel[2:]

    freq_res = nyquist_hertz / float(num_spectrogram_bins)
    freq_th = 1.5 * freq_res
    for i in range(0, num_mel_bins):
        center_hz = mel_to_hertz(center_mel[i])
        lower_hz = mel_to_hertz(lower_edge_mel[i])
        upper_hz = mel_to_hertz(upper_edge_mel[i])
        if upper_hz - lower_hz < freq_th:
            rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
            dm = _MEL_HIGH_FREQUENCY_Q * np.log(rhs + np.sqrt(1.0 + rhs**2))
            lower_edge_mel[i] = center_mel[i] - dm
            upper_edge_mel[i] = center_mel[i] + dm

    lower_edge_hz = mel_to_hertz(lower_edge_mel)[np.newaxis, :]
    center_hz = mel_to_hertz(center_mel)[np.newaxis, :]
    upper_edge_hz = mel_to_hertz(upper_edge_mel)[np.newaxis, :]

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (linear_frequencies - lower_edge_hz) / (
        center_hz - lower_edge_hz)
    upper_slopes = (upper_edge_hz - linear_frequencies) / (
        upper_edge_hz - center_hz)

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    # [freq, mel]
    mel_weights_matrix = np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]],
                                'constant')
    return mel_weights_matrix


def _mel_to_linear_matrix(*args, **kwargs):
    """Get the inverse mel transformation matrix."""
    m = _linear_to_mel_matrix(*args, **kwargs)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def diff(inputs, axis=-1):
    size = inputs.shape.as_list()
    size[axis] -= 1
    begin_back = [0] * len(size)
    begin_front = [0] * len(size)
    begin_front[axis] = 1
    slice_back = tf.slice(inputs, begin_back, size)
    slice_front = tf.slice(inputs, begin_front, size)
    diffs = slice_front - slice_back
    return diffs


def unwrap(phases, axis=-1):
    diffs = diff(phases, axis=axis)
    mods = tf.mod(diffs + np.pi, np.pi * 2.0) - np.pi
    indices = tf.logical_and(tf.equal(mods, -np.pi), tf.greater(diffs, 0.0))
    mods = tf.where(indices, tf.ones_like(mods) * np.pi, mods)
    corrects = mods - diffs
    cumsums = tf.cumsum(corrects, axis=axis)
    shape = phases.shape.as_list()
    shape[axis] = 1
    cumsums = tf.concat([tf.zeros(shape), cumsums], axis=axis)
    unwrapped = phases + cumsums
    return unwrapped


def instantaneous_frequency(phases, axis=-2):
    unwrapped = unwrap(phases, axis=axis)
    diffs = diff(unwrapped, axis=axis)
    size = unwrapped.shape.as_list()
    size[axis] = 1
    begin = [0] * len(size)
    initials = tf.slice(unwrapped, begin, size)
    diffs = tf.concat([initials, diffs], axis=axis) / np.pi
    return diffs


def convert_to_spectrogram(waveforms, waveform_length, sample_rate, spectrogram_shape, overlap):

    def normalize(inputs, mean, stddev):
        return (inputs - mean) / stddev

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1.0 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    waveforms = tf.pad(waveforms, [[0, 0], [num_samples - waveform_length, 0]])

    stfts = tf.signal.stft(
        signals=waveforms,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=functools.partial(
            tf.signal.hann_window,
            periodic=True
        )
    )
    # discard_dc
    stfts = stfts[..., 1:]

    magnitude_spectrograms = tf.abs(stfts)
    phase_spectrograms = tf.angle(stfts)

    # this matrix can be constant by graph optimization `Constant Folding`
    # since there are no Tensor inputs
    linear_to_mel_weight_matrix = tf.cast(_linear_to_mel_weight_matrix(
        num_mel_bins=num_freq_bins,
        num_spectrogram_bins=num_freq_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate / 2.0
    ), tf.float32)
    mel_magnitude_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, axes=1)
    mel_magnitude_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    mel_phase_spectrograms = tf.tensordot(phase_spectrograms, linear_to_mel_weight_matrix, axes=1)
    mel_phase_spectrograms.set_shape(phase_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_magnitude_spectrograms = tf.log(mel_magnitude_spectrograms + 1.0e-6)
    mel_instantaneous_frequencies = instantaneous_frequency(mel_phase_spectrograms, axis=-2)

    log_mel_magnitude_spectrograms = normalize(log_mel_magnitude_spectrograms, -3.76, 10.05)
    mel_instantaneous_frequencies = normalize(mel_instantaneous_frequencies, 0.0, 1.0)

    return log_mel_magnitude_spectrograms, mel_instantaneous_frequencies


def convert_to_waveform(log_mel_magnitude_spectrograms, mel_instantaneous_frequencies, waveform_length, sample_rate, spectrogram_shape, overlap):

    def unnormalize(inputs, mean, stddev):
        return inputs * stddev + mean

    time_steps, num_freq_bins = spectrogram_shape
    frame_length = num_freq_bins * 2
    frame_step = int((1.0 - overlap) * frame_length)
    num_samples = frame_step * (time_steps - 1) + frame_length

    log_mel_magnitude_spectrograms = unnormalize(log_mel_magnitude_spectrograms, -3.76, 10.05)
    mel_instantaneous_frequencies = unnormalize(mel_instantaneous_frequencies, 0.0, 1.0)

    mel_magnitude_spectrograms = tf.exp(log_mel_magnitude_spectrograms)
    mel_phase_spectrograms = tf.cumsum(mel_instantaneous_frequencies * np.pi, axis=-2)

    # this matrix can be constant by graph optimization `Constant Folding`
    # since there are no Tensor inputs
    mel_to_linear_weight_matrix = tf.cast(_mel_to_linear_weight_matrix(
        num_mel_bins=num_freq_bins,
        num_spectrogram_bins=num_freq_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate / 2.0
    ), tf.float32)
    magnitudes = tf.tensordot(mel_magnitude_spectrograms, mel_to_linear_weight_matrix, axes=1)
    magnitudes.set_shape(mel_magnitude_spectrograms.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
    phase_spectrograms = tf.tensordot(mel_phase_spectrograms, mel_to_linear_weight_matrix, axes=1)
    phase_spectrograms.set_shape(mel_phase_spectrograms.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))

    stfts = tf.complex(magnitudes, 0.0) * tf.complex(tf.cos(phase_spectrograms), tf.sin(phase_spectrograms))

    # discard_dc
    stfts = tf.pad(stfts, [[0, 0], [0, 0], [1, 0]])
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

    # For Nsynth dataset, we are putting all padding in the front
    # This causes edge effects in the tail
    waveforms = waveforms[:, num_samples - waveform_length:]

    return waveforms


def cross_correlation(x, y, padding="VALID", normalize=True):

    if normalize:
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)

    x = x[..., tf.newaxis, tf.newaxis]
    y = y[..., tf.newaxis, tf.newaxis]

    cross_correlations = tf.map_fn(
        fn=lambda inputs: tf.squeeze(tf.nn.conv2d(
            input=inputs[0][tf.newaxis, ...],
            filter=inputs[1][..., tf.newaxis],
            strides=[1, 1, 1, 1],
            padding=padding,
            data_format="NHWC",
        )),
        elems=(x, y),
        dtype=tf.float32,
        swap_memory=True
    )

    return cross_correlations



class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, spectral_params, hyper_params):

        # -----------------------------------------------------------------------------------------
        # Non-Saturating Loss + Mode-Seeking Loss + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis]
        # (https://arxiv.org/pdf/1903.05628.pdf)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------

        real_waveforms, labels = real_input_fn()
        latents = fake_input_fn()

        fake_images = generator(latents, labels)

        real_magnitude_spectrograms, real_instantaneous_frequencies = convert_to_spectrogram(real_waveforms, **spectral_params)
        real_images = tf.stack([real_magnitude_spectrograms, real_instantaneous_frequencies], axis=1)

        fake_magnitude_spectrograms, fake_instantaneous_frequencies = tf.unstack(fake_images, axis=1)
        fake_waveforms = convert_to_waveform(fake_magnitude_spectrograms, fake_instantaneous_frequencies, **spectral_params)

        real_logits = discriminator(real_images, labels)
        fake_logits = discriminator(fake_images, labels)

        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += real_gradient_penalties * hyper_params.real_gradient_penalty_weight
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            fake_gradients = tf.gradients(fake_logits, [fake_images])[0]
            fake_gradient_penalties = tf.reduce_sum(tf.square(fake_gradients), axis=[1, 2, 3])
            discriminator_losses += fake_gradient_penalties * hyper_params.fake_gradient_penalty_weight

        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits)
        # gradient-based mode-seeking loss
        if hyper_params.mode_seeking_loss_weight:
            latent_gradients = tf.gradients(fake_images, [latents])[0]
            mode_seeking_losses = 1.0 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1.0e-6)
            generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight

        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)

        generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )

        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )

        self.real_waveforms = real_waveforms
        self.fake_waveforms = fake_waveforms
        self.real_magnitude_spectrograms = real_magnitude_spectrograms
        self.fake_magnitude_spectrograms = fake_magnitude_spectrograms
        self.real_instantaneous_frequencies = real_instantaneous_frequencies
        self.fake_instantaneous_frequencies = fake_instantaneous_frequencies
        self.real_images = real_images
        self.fake_images = fake_images
        self.real_labels = labels
        self.fake_labels = labels
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_train_op = generator_train_op
        self.discriminator_train_op = discriminator_train_op

    def train(self, model_dir, config, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    ),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.audio(
                            name=name,
                            tensor=tensor,
                            sample_rate=16000,
                            max_outputs=4
                        ) for name, tensor in dict(
                            real_waveforms=self.real_waveforms,
                            fake_waveforms=self.fake_waveforms
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.image(
                            name=name,
                            tensor=tensor,
                            max_outputs=4
                        ) for name, tensor in dict(
                            real_magnitude_spectrograms=self.real_magnitude_spectrograms[..., tf.newaxis],
                            fake_magnitude_spectrograms=self.fake_magnitude_spectrograms[..., tf.newaxis],
                            real_instantaneous_frequencies=self.real_instantaneous_frequencies[..., tf.newaxis],
                            fake_instantaneous_frequencies=self.fake_instantaneous_frequencies[..., tf.newaxis]
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(
                            name=name,
                            tensor=tensor
                        ) for name, tensor in dict(
                            generator_loss=self.generator_loss,
                            discriminator_loss=self.discriminator_loss
                        ).items()
                    ]),
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        generator_loss=self.generator_loss,
                        discriminator_loss=self.discriminator_loss
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                try:
                    session.run(self.discriminator_train_op)
                    session.run(self.generator_train_op)
                except tf.errors.OutOfRangeError:
                    break

    def evaluate(self, model_dir, config, classifier, images, features, logits):

        real_features, real_logits = tf.import_graph_def(
            graph_def=classifier,
            input_map={images: self.real_images},
            return_elements=[features, logits]
        )

        fake_features, fake_logits = tf.import_graph_def(
            graph_def=classifier,
            input_map={images: self.fake_images},
            return_elements=[features, logits]
        )

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            def generator():
                while not session.should_stop():
                    try:
                        yield session.run([real_features, real_logits, fake_features, fake_logits])
                    except tf.errors.OutOfRangeError:
                        break

            real_features, real_logits, fake_features, fake_logits = map(np.concatenate, zip(*generator()))

            return dict(
                frechet_inception_distance=metrics.frechet_inception_distance(real_features, fake_features),
                real_inception_score=metrics.inception_score(real_logits),
                fake_inception_score=metrics.inception_score(fake_logits),
                num_different_bins=metrics.num_different_bins(real_features, fake_features)
            )

    def generate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            while not session.should_stop():
                try:
                    yield session.run(self.fake_waveforms)
                except tf.errors.OutOfRangeError:
                    break


class PitchClassifier(object):

    def __init__(self, network, input_fn, spectral_params, hyper_params):

        waveforms, labels = input_fn()

        magnitude_spectrograms, instantaneous_frequencies = spectral_ops.convert_to_spectrogram(waveforms, **spectral_params)
        images = tf.stack([magnitude_spectrograms, instantaneous_frequencies], axis=1)

        features, logits = network(images)

        loss = tf.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=labels
        )
        loss += tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
            if "normalization" not in variable.name
        ]) * hyper_params.weight_decay

        accuracy, update_op = tf.metrics.accuracy(
            predictions=tf.argmax(logits, axis=-1),
            labels=tf.argmax(labels, axis=-1)
        )

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=(
                hyper_params.learning_rate(tf.train.get_or_create_global_step())
                if callable(hyper_params.learning_rate) else hyper_params.learning_rate
            ),
            momentum=hyper_params.momentum,
            use_nesterov=hyper_params.use_nesterov
        )
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step()
            )

        self.waveforms = waveforms
        self.magnitude_spectrograms = magnitude_spectrograms
        self.instantaneous_frequencies = instantaneous_frequencies
        self.loss = loss
        self.accuracy = accuracy
        self.train_op = train_op
        self.update_op = update_op

        images = tf.placeholder(tf.float32, shape=[None, *images.shape[1:]], name="images")
        features, logits = network(images)
        features = tf.identity(features, name="features")
        logits = tf.identity(logits, name="logits")

    def train(self, model_dir, config, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    ),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.audio(
                            name=name,
                            tensor=tensor,
                            sample_rate=16000,
                            max_outputs=4
                        ) for name, tensor in dict(
                            waveforms=self.waveforms
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.image(
                            name=name,
                            tensor=tensor,
                            max_outputs=4
                        ) for name, tensor in dict(
                            magnitude_spectrograms=self.magnitude_spectrograms[..., tf.newaxis],
                            instantaneous_frequencies=self.instantaneous_frequencies[..., tf.newaxis]
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(
                            name=name,
                            tensor=tensor
                        ) for name, tensor in dict(
                            loss=self.loss,
                            accuracy=self.accuracy
                        ).items()
                    ]),
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        loss=self.loss,
                        accuracy=self.accuracy
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                try:
                    session.run([self.train_op, self.update_op])
                except tf.errors.OutOfRangeError:
                    break

    def evaluate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            while not session.should_stop():
                try:
                    accuracy = session.run(self.update_op)
                except tf.errors.OutOfRangeError:
                    break

            return dict(accuracy=accuracy)
