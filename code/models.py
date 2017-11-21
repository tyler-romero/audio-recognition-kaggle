import tensorflow as tf
from tensorflow import layers

# TODO: Just try a lot of different things.
# For reference, SimpleConv is about half the maximum model size


def create_model(FLAGS):
    if FLAGS.model_architecture == "Baseline":
        return Baseline(FLAGS)
    if FLAGS.model_architecture == "SimpleConv":
        return SimpleConv(FLAGS)
    if FLAGS.model_architecture == "BiggerConv":
        return BiggerConv(FLAGS)
    else:
        raise Exception("Invalid model architecture")


class Model():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def forward(self, X, is_training):
        raise Exception("NotImplementedError")

    def loss(self, y):
        """
        Setup default loss after setting up the forward pass.
        This is the standard softmax cross entropy loss function
        Must save the output as self.raw_output during the forward pass
        y: the correct labels

        - Returns:
        loss: a double
        """
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=self.raw_scores
        )
        loss = tf.reduce_mean(l)
        return loss

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)   # Recommended lr of 1e-3

    def train_op(self, lr, step, loss):
        """
        Setup default optimizer after setting up loss
        lr: learning rate
        step: global step

        - Returns:
        train_op: a handle on the training operation
        decayed_lr: the current lr after exponential decay
        global_norm: the current global_norm
        """

        optimizer = self.get_optimizer(lr)

        grads_and_vars = optimizer.compute_gradients(
            loss, tf.trainable_variables()
        )
        grads = [g for g, v in grads_and_vars]
        global_norm = tf.global_norm(grads)

        # Batch Norm in tensorflow requires this extra dependency
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=step
            )

        return train_op, global_norm


#############################################################################

class Baseline(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward(self, x, is_training):
        x = layers.batch_normalization(x, training=is_training)
        x = layers.dense(inputs=x, units=100, activation=tf.nn.relu)
        self.raw_scores = layers.dense(
            inputs=x, units=self.FLAGS.num_classes, activation=None
        )
        return self.raw_scores


# Modified from CNNs for Small-footprint Keyword Spotting
class SimpleConv(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward(self, x, is_training):
        # Assumes [Batch, Time, Freq, Chan]
        x = layers.conv2d(
            x, filters=64, kernel_size=[20, 8], strides=1,
            activation=tf.nn.relu
        )
        x = layers.max_pooling2d(    # Pool over frequency only
            x, pool_size=[1, 3], strides=[1, 3]
        )
        x = layers.conv2d(
            x, filters=64, kernel_size=[10, 4], strides=1,
            activation=tf.nn.relu
        )

        x = tf.contrib.layers.flatten(x)
        self.raw_scores = layers.dense(  # Linear map to output
            inputs=x, units=self.FLAGS.num_classes, activation=None
        )
        return self.raw_scores


class BiggerConv(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward(self, x, is_training):
        # Assumes [Batch, Time, Freq, Chan]
        x = layers.conv2d(
            x, filters=128, kernel_size=[20, 8], strides=1,
            activation=tf.nn.relu
        )
        x = layers.max_pooling2d(    # Pool over frequency only
            x, pool_size=[1, 3], strides=[1, 3]
        )
        x = layers.conv2d(
            x, filters=128, kernel_size=[10, 4], strides=1,
            activation=tf.nn.relu
        )

        x = tf.contrib.layers.flatten(x)
        self.raw_scores = layers.dense(  # Linear map to output
            inputs=x, units=self.FLAGS.num_classes, activation=None
        )
        return self.raw_scores

