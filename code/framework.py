
import random
import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from utils import get_batches


class Framework():
    def __init__(self, sess, model, experiment, FLAGS):
        self.sess = sess
        self.model = model
        self.experiment = experiment
        self.FLAGS = FLAGS

        # Setup instance variables
        self.global_step = tf.Variable(int(0), trainable=False, name="global_step")

        # Set up placeholders
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.X = tf.placeholder(tf.float32, [None, FLAGS.sample_rate], name="X")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # Assemble pieces
        with tf.variable_scope("model"):
            self.setup_system()
            self.setup_loss()
            self.setup_training_procedure()

        # Initialize all variables/parameters
        init = tf.global_variables_initializer()
        sess.run(init)
        self.experiment.set_model_graph(sess.graph)

    # Setup the specified loss function
    def setup_loss(self):
        with vs.variable_scope("loss"):
            self.loss = self.model.loss(self.y)

    # Set up the forward pass
    def setup_system(self):
        with vs.variable_scope("classify"):
            raw_scores = self.model.forward(self.X, self.is_training)
            self.y_out = tf.nn.softmax(raw_scores, name="softmax")

    # Set up the optimization step
    def setup_training_procedure(self):
        with vs.variable_scope("train_op"):
            self.train_op, self.global_norm = self.model.train_op(
                self.learning_rate,
                self.global_step,
                self.loss
            )

    # Classify given the features
    def classify(self, X_batch):
        input_feed = {}
        input_feed[self.X] = X_batch
        input_feed[self.is_training] = False

        output_feed = [self.y_out]
        outputs = self.sess.run(output_feed, input_feed)
        outputs = outputs[0]  # Run returns the output feed as a list. We just want the first element

        preds = np.argmax(np.array(outputs), axis=1)
        return preds

    # Calculate the current accuracy of the model
    def evaluate(self, X, y, sample_size=None):
        dataset = list(zip(X, y))
        if sample_size is None:
            sample_size = len(dataset)

        eval_set = random.sample(dataset, sample_size)

        running_sum = 0
        for wav, label in eval_set:
            wav = np.expand_dims(wav, axis=0)
            pred = self.classify(wav)
            correct_pred = np.equal(pred, label)
            running_sum += np.sum(correct_pred)

        accuracy = running_sum / sample_size
        return accuracy

    # Perform a single batched update step
    def step(self, X_batch, y_batch):
        """
        FOR TRAINING ONLY
        
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
            loss, global_norm, global_step
        """
        input_feed = {}
        input_feed[self.X] = X_batch
        input_feed[self.y] = y_batch
        input_feed[self.is_training] = True
        input_feed[self.learning_rate] = self.FLAGS.learning_rate

        output_feed = []
        output_feed.append(self.train_op)
        output_feed.append(self.loss)
        output_feed.append(self.global_norm)
        output_feed.append(self.global_step)

        tr, loss, norm, step = self.sess.run(output_feed, input_feed)
        return loss, norm, step

    # The complete optimization function that fits the model
    def optimize(self, X_train, y_train, X_val, y_val):
        # Helper stuff
        num_data = len(X_train)

        # Epoch level loop
        step = 1
        for cur_epoch in range(self.FLAGS.epochs):
            X_batches, y_batches, num_batches = get_batches(
                X_train, y_train, self.FLAGS.batch_size
            )

            # Training loop
            for _i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
                i = _i + 1  # For convienince

                # Optimatize using batch
                loss, norm, step = self.step(X_batch, y_batch)
                self.experiment.log_loss(loss)

                # Print relevant params
                num_complete = int(20 * (self.FLAGS.batch_size*i/num_data))
                sys.stdout.write('\r')
                sys.stdout.write("EPOCH: %d ==> (Batch Loss: %.3f) [%-20s] (%d/%d) [norm: %.2f] [step: %d]"
                    % (cur_epoch + 1, loss, '=' * num_complete, min(i * self.FLAGS.batch_size, num_data), num_data, norm, step))
                sys.stdout.flush()

                self.experiment.log_step(int(step))

            sys.stdout.write('\n')

            # Evaluate accuracy
            eval_size = min(len(X_val), len(X_train)) // 10

            train_acc = self.evaluate(X_train, y_train, eval_size)
            print("Training Accuracy: {}\t\ton {} examples".format(train_acc, eval_size))
            self.experiment.log_metric("train_acc", train_acc)

            val_acc = self.evaluate(X_val, y_val, eval_size)
            print("Validation Accuracy: {}\ton {} examples".format(val_acc, eval_size))
            self.experiment.log_accuracy(val_acc)