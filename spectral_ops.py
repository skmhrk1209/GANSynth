import tensorflow as tf
import numpy as np


def diff(inputs, axis=-1):
    """Take the finite difference of a tensor along an axis.
    Args:
      x: Input tensor of any dimension.
      axis: Axis on which to take the finite difference.
    Returns:
      d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
      ValueError: Axis out of range for tensor.
    """
    begin_back = [0] * inputs.shape.ndims
    begin_front = [0] * inputs.shape.ndims
    begin_front[axis] = 1

    size = [-1] * inputs.shape.ndims
    size[axis] = inputs.shape[axis].value - 1

    front = tf.slice(inputs, begin_front, size)
    back = tf.slice(inputs, begin_back, size)

    return front - back


def unwrap(phases, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
      p: Phase tensor.
      discont: Float, size of the cyclic discontinuity.
      axis: Axis of which to unwrap.
    Returns:
      unwrapped: Unwrapped tensor of same size as input.
    """
    diffs = diff(phases, axis=axis)
    diff_mods = tf.mod(diffs + np.pi, 2.0 * np.pi) - np.pi
    indices = tf.logical_and(tf.equal(diff_mods, -np.pi), tf.greater(diffs, 0))
    diff_mods = tf.where(indices, tf.ones_like(diff_mods) * np.pi, diff_mods)
    corrects = diff_mods - diffs
    cumsums = tf.cumsum(corrects, axis=axis)

    shape = phases.shape
    shape.dims[0] = tf.shape(phases)[0]
    shape.dims[axis] = 1

    cumsums = tf.concat([tf.zeros(shape), cumsums], axis=axis)

    return phases + cumsums


def instantaneous_frequency(phases, time_axis=-2):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.
    Args:
      phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
      time_axis: Axis over which to unwrap and take finite difference.
    Returns:
      dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    unwrapped = unwrap(phases, axis=time_axis)
    diffs = diff(unwrapped, axis=time_axis)

    begin = [0] * unwrapped.shape.ndims

    size = [-1] * unwrapped.shape.ndims
    size[time_axis] = 1

    unwrapped = tf.slice(unwrapped, begin, size)
    diffs = tf.concat([unwrapped, diffs], axis=time_axis) / np.pi

    return diffs
