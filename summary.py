import tensorflow as tf
import numpy as np
import re


def scalar(tensor, name=None, **kwargs):

    name = name or re.sub(":.*", "", tensor.name)
    tf.summary.scalar(name, tensor, **kwargs)


def image(tensor, name=None, **kwargs):

    name = name or re.sub(":.*", "", tensor.name)
    tensor = tf.cond(
        pred=tf.equal(tf.argmin(tf.shape(tensor)[1:]), 0),
        true_fn=lambda: tf.transpose(tensor, [0, 2, 3, 1]),
        false_fn=lambda: tensor
    )
    tf.summary.image(name, tensor, **kwargs)
