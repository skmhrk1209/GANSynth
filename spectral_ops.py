import tensorflow as tf
import numpy as np


def diff(x, axis=-1):
    """Take the finite difference of a tensor along an axis.
    Args:
      x: Input tensor of any dimension.
      axis: Axis on which to take the finite difference.
    Returns:
      d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
      ValueError: Axis out of range for tensor.
    """
    shape = x.get_shape()
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                         (axis, len(shape)))

    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]
    begin_front[axis] = 1

    size = shape.as_list()
    size[axis] -= 1
    slice_front = tf.slice(x, begin_front, size)
    slice_back = tf.slice(x, begin_back, size)
    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
      p: Phase tensor.
      discont: Float, size of the cyclic discontinuity.
      axis: Axis of which to unwrap.
    Returns:
      unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    idx = tf.less(tf.abs(dd), discont)
    ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = tf.cumsum(ph_correct, axis=axis)

    shape = p.get_shape().as_list()
    shape[axis] = 1
    ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    return unwrapped


def instantaneous_frequency(phase_angle, time_axis=-2):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.
    Args:
      phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
      time_axis: Axis over which to unwrap and take finite difference.
    Returns:
      dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)
    dphase = diff(phase_unwrapped, axis=time_axis)

    # Add an initial phase to dphase
    size = phase_unwrapped.get_shape().as_list()
    size[time_axis] = 1
    begin = [0 for unused_s in size]
    phase_slice = tf.slice(phase_unwrapped, begin, size)
    dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase
