import tensorflow as tf
import numpy as np


def difference(input, axis=-1):
    """ Take the finite difference of a tensor along an axis.
    Refference:
      https://github.com/numpy/numpy/blob/v1.15.0/numpy/lib/function_base.py#L1355-L1408
    Args:
      input: Input tensor of any dimension.
      axis: Axis on which to take the finite difference.
    Returns:
      diff: Tensor with size less than input by 1 along the difference dimension.
    Raises:
      ValueError: Axis out of range for tensor.
    """
    if axis >= len(input.shape):
        raise ValueError(
            "Invalid axis index: {} for tensor "
            "with only {} axes.".format(axis, len(input.shape))
        )

    size = input.shape.as_list()
    size[axis] -= 1
    begin_back = [0] * len(size)
    begin_front = [0] * len(size)
    begin_back[axis] = 0
    begin_front[axis] = 1
    input_front = tf.slice(input, begin_front, size)
    input_back = tf.slice(input, begin_back, size)
    diff = input_front - input_back
    return diff


def unwrap(phase, discont=np.pi, axis=-1):
    """ Unwrap a cyclical phase tensor.
    Refference:
      https://github.com/numpy/numpy/blob/v1.15.0/numpy/lib/function_base.py#L1355-L1408
    Args:
      phase: Phase tensor.
      discont: Float, size of the cyclic discontinuity.
      axis: Axis of which to unwrap.
    Returns:
      unwrapped: Unwrapped tensor of same size as input.
    """
    diff = difference(phase, axis=axis)
    diff_mod = tf.mod(diff + np.pi, 2.0 * np.pi) - np.pi
    indices = tf.logical_and(tf.equal(diff_mod, -np.pi), tf.greater(diff, 0))
    diff_mod = tf.where(indices, tf.ones_like(diff_mod) * np.pi, diff_mod)
    phase_correct = diff_mod - diff
    phase_cumsum = tf.cumsum(phase_correct, axis=axis)
    shape = phase.shape.as_list()
    shape[axis] = 1
    phase_cumsum = tf.concat([tf.zeros(shape, dtype=phase.dtype), phase_cumsum], axis=axis)
    unwrapped = phase + phase_cumsum
    return unwrapped


def instantaneous_frequency(phase, time_axis=-2):
    """ Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.
    Refference:
      https://github.com/numpy/numpy/blob/v1.15.0/numpy/lib/function_base.py#L1355-L1408
    Args:
      phase: Tensor of angles in radians. [time, freqs]
      time_axis: Axis over which to unwrap and take finite difference.
    Returns:
      diff: Instantaneous frequency (derivative of phase). Same size as input.
    """
    unwrapped = unwrap(phase, axis=time_axis)
    diff = difference(unwrapped, axis=time_axis)
    # Add an initial phase to dphase
    size = unwrapped.shape.as_list()
    size[time_axis] = 1
    begin = [0] * len(size)
    unwrapped = tf.slice(unwrapped, begin, size)
    diff = tf.concat([unwrapped, diff], axis=time_axis) / np.pi
    return diff
