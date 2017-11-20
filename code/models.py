import tensorflow as tf
from tensorflow import layers


def create_model(FLAGS):
    if FLAGS.model_architecture == "Baseline":
        return Baseline(FLAGS)
    if FLAGS.model_architecture == "SimpleConv":
        return SimpleConv(FLAGS)
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
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.raw_scores)
        loss = tf.reduce_mean(l)
        return loss

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)   # Recommended lr of 1e-3

    def train_op(self, lr, step, loss):
        """
        Setup default optimizer after setting up loss
        This is the standard training operation with Adam optimizer, decayed lr, and gradient clipping
        lr: learning rate
        step: global step

        - Returns:
        train_op: a handle on the training operation
        decayed_lr: the current lr after exponential decay
        global_norm: the current global_norm
        """

        optimizer = self.get_optimizer(lr)

        grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
        grads = [g for g, v in grads_and_vars]
        global_norm = tf.global_norm(grads)

        # Batch Norm in tensorflow requires this extra dependency
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=step)

        return train_op, global_norm


#############################################################################

class Baseline(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward(self, x, is_training):
        x = layers.batch_normalization(x, training=is_training)
        x = layers.dense(inputs=x, units=100, activation=tf.nn.relu)
        self.raw_scores = layers.dense(inputs=x, units=self.FLAGS.num_classes, activation=None)
        return self.raw_scores


class SimpleConv(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward(self, x, is_training):
        x = layers.batch_normalization(x, training=is_training)  # Automatically standardize inputs

        x = tf.expand_dims(x, -1)  # Add a channel dimension

        x = layers.conv1d(x, filters=10, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
        x = layers.conv1d(x, filters=20, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
        x = layers.max_pooling1d(x, pool_size=2, strides=2)
        x = layers.conv1d(x, filters=30, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
        x = layers.conv1d(x, filters=40, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
        x = layers.max_pooling1d(x, pool_size=2, strides=2)

        x = tf.contrib.layers.flatten(x)

        self.raw_scores = layers.dense(inputs=x, units=self.FLAGS.num_classes, activation=None)
        return self.raw_scores
