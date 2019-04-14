import tensorflow as tf
import numpy as np
import time


tf.logging.set_verbosity(tf.logging.INFO)


def batch_normalization(inputs, training, decay=0.99, epsilon=0.001):
    training = tf.convert_to_tensor(training)
    shape = inputs.shape.as_list()
    mean = tf.get_variable(
        name="mean",
        shape=[1, shape[1]] + [1] * len(shape[2:]),
        initializer=tf.initializers.zeros(),
        trainable=False
    )
    variance = tf.get_variable(
        name="variance",
        shape=[1, shape[1]] + [1] * len(shape[2:]),
        initializer=tf.initializers.ones(),
        trainable=False
    )
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    ema_apply_op = ema.apply([mean, variance])
    batch_mean, batch_variance = tf.nn.moments(
        x=inputs,
        axes=[0] + list(range(2, len(shape))),
        keep_dims=True
    )
    moving_mean = ema.average(mean)
    moving_variance = ema.average(variance)
    mean_assign_op = tf.assign(mean, batch_mean)
    variance_assign_op = tf.assign(variance, batch_variance)
    with tf.control_dependencies([mean_assign_op, variance_assign_op]):
        with tf.control_dependencies([tf.cond(training, lambda: ema_apply_op, lambda: tf.no_op())]):
            mean = tf.cond(training, lambda: batch_mean, lambda: moving_mean)
            variance = tf.cond(training, lambda: batch_variance, lambda: moving_variance)
            stddev = tf.sqrt(variance + epsilon)
            inputs = (inputs - mean) / stddev
            beta = tf.get_variable(
                name="beta",
                shape=[1, shape[1]] + [1] * len(shape[2:]),
                initializer=tf.initializers.zeros()
            )
            gamma = tf.get_variable(
                name="gamma",
                shape=[1, shape[1]] + [1] * len(shape[2:]),
                initializer=tf.initializers.ones()
            )
            inputs = inputs * gamma + beta
    return inputs


def conv_net(features, labels, mode):

    inputs = tf.reshape(
        tensor=features["images"],
        shape=[-1, 28, 28, 1]
    )
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        data_format="channels_first"
    )
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        data_format="channels_first"
    )
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        data_format="channels_first"
    )
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        data_format="channels_first"
    )
    inputs = tf.layers.flatten(inputs)
    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024,
        activation=tf.nn.relu
    )
    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )
    predictions = tf.argmax(
        input=logits,
        axis=1
    )
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=tf.train.AdamOptimizer().minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                accuracy=tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions
                )
            )
        )


if __name__ == "__main__":

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = np.asarray(mnist.train.images, dtype=np.float32)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = np.asarray(mnist.test.images, dtype=np.float32)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=conv_net,
        model_dir="mnist_convnet_model",
        config=tf.estimator.RunConfig(
            save_summary_steps=1000,
            save_checkpoints_steps=1000,
            log_step_count_steps=100
        )
    )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=10,
        shuffle=True
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": eval_data},
        y=eval_labels,
        batch_size=100,
        num_epochs=1,
        shuffle=False
    )

    begin = time.time()
    mnist_classifier.train(train_input_fn)
    print(mnist_classifier.evaluate(eval_input_fn))
    end = time.time()

    print("elapsed_time: {}s".format(end - begin))
