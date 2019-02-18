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

    begin_back = [0] * len(inputs.shape)
    begin_front = [0] * len(inputs.shape)
    begin_front[axis] = 1

    size = [-1] * len(inputs.shape)
    size[axis] = inputs.shape[axis].value - 1

    slice_front = tf.slice(inputs, begin_front, size)
    slice_back = tf.slice(inputs, begin_back, size)
    diff = slice_front - slice_back

    return diff


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
    phase_corrects = diff_mods - diffs  
    phase_cumsums = tf.cumsum(phase_corrects, axis=axis)

    shape = phases.shape.as_list()
    shape[axis] = 1

    phase_cumsums = tf.concat([tf.zeros(shape), phase_cumsums], axis=axis)
    unwrapped = phases + phase_cumsums

    return unwrapped


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

    # Add an initial phase to dphase
    begin = [0] * len(unwrapped.shape)

    size = [-1] * len(unwrapped.shape)
    size[time_axis] = 1

    phase_slice = tf.slice(unwrapped, begin, size)
    diffs = tf.concat([phase_slice, diffs], axis=time_axis) / np.pi
    
    return diffs
