import tensorflow as tf
import numpy as np
import re


def scalar(tensor, name=None, **kwargs):

    name = name or re.sub(":.*", "", tensor.name)
    tf.summary.scalar(name, tensor, **kwargs)


def image(tensor, name=None, **kwargs):

    name = name or re.sub(":.*", "", tensor.name)
    if np.argmin(tensor.shape.as_list()[1:]) == 0:
        tensor = tf.transpose(tensor, [0, 2, 3, 1])
    tf.summary.image(name, tensor, **kwargs)
