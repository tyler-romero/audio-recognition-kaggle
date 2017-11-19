import tensorflow as tf
from tensorflow.contrib import layers


def create_model(FLAGS):
    if FLAGS.architecture == "Baseline":
        return Baseline(FLAGS)
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
        regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print("Number of regularizers: ", len(regs))
        reg_loss = self.FLAGS.weight_decay * tf.reduce_sum(regs)
        return loss + reg_loss

    def get_optimizer(self, lr):
        if self.FLAGS.optimizer == "adam":
            return tf.train.AdamOptimizer(lr)   # Recommended lr of 1e-3
        elif self.FLAGS.optimizer == 'nesterov':   
            return tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True)   # Recommended lr of 0.1
        elif self.FLAGS.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(lr)    # Recommended lr of 1e-2
        else:
            raise Exception("InvalidOptimizerError")

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
        variables = [v for g, v in grads_and_vars]

        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)

        # Batch Norm in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step=step, name="apply_clipped_grads")

        return train_op, global_norm


#############################################################################

class Baseline(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward():

