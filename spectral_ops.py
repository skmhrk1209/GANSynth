import tensorflow as tf
import numpy as np


def diff(inputs, axis=-1):

    begin_back = [0] * inputs.shape.ndims
    begin_front = [0] * inputs.shape.ndims
    begin_front[axis] = 1

    size = [-1] * inputs.shape.ndims
    size[axis] = inputs.shape[axis].value - 1

    front = tf.slice(inputs, begin_front, size)
    back = tf.slice(inputs, begin_back, size)

    return front - back


def unwrap(phases, discont=np.pi, axis=-1):

    diffs = diff(phases, axis=axis)
    diff_mods = tf.mod(diffs + np.pi, 2.0 * np.pi) - np.pi
    indices = tf.logical_and(tf.equal(diff_mods, -np.pi), tf.greater(diffs, 0))
    diff_mods = tf.where(indices, tf.ones_like(diff_mods) * np.pi, diff_mods)
    corrects = diff_mods - diffs
    cumsums = tf.cumsum(corrects, axis=axis)

    shape = phases.shape.as_list()
    shape[0] = tf.shape(phases)[0]
    shape[axis] = 1

    cumsums = tf.concat([tf.zeros(shape), cumsums], axis=axis)

    return phases + cumsums


def instantaneous_frequency(phases, axis=-2):

    unwrapped = unwrap(phases, axis=axis)
    diffs = diff(unwrapped, axis=axis)

    begin = [0] * unwrapped.shape.ndims

    size = [-1] * unwrapped.shape.ndims
    size[axis] = 1

    unwrapped = tf.slice(unwrapped, begin, size)
    diffs = tf.concat([unwrapped, diffs], axis=axis) / np.pi

    return diffs
