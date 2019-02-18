import tensorflow as tf
import re


def scalar(tensor, name=None, **kwargs):

    name = name or re.sub(":.*", "", tensor.name)
    tf.summary.scalar(name, tensor, **kwargs)


def image(tensor, name=None, data_format="channels_first", **kwargs):

    name = name or re.sub(":.*", "", tensor.name)
    if data_format == "channels_first":
        tensor = tf.transpose(tensor, [0, 2, 3, 1])
    tf.summary.image(name, tensor, **kwargs)
