import tensorflow as tf


class Experiment():
    def __init__(self, sess, model, FLAGS):
        self.sess = sess
        self.model = model
        self.FLAGS = FLAGS

        # Setup instance variables
        self.global_step = tf.Variable(int(0), trainable=False, name="step")
        self.learning_rate = self.FLAGS.learning_rate

        # Assemble pieces
        with tf.variable_scope("model"):
            self.setup_loss()
            self.setup_system()
            self.setup_training_procedure()
     
    def setup_loss():
        with vs.variable_scope("loss"):
            self.loss = self.model.loss(self.y)

    def setup_system():
        with vs.variable_scope("classify"):
            raw_scores = self.model.forward_pass(self.X, self.is_training)
            self.y_out = tf.nn.softmax(raw_scores, name="softmax")

            with tf.name_scope('y_out_summaries'):
                mean = tf.reduce_mean(self.y_out)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(self.y_out - mean)))

    def setup_training_procedure():
        with vs.variable_scope("train_op"):
            self.train_op, self.global_norm = self.model.train_op(
                self.learning_rate, self.global_step, self.loss
            )

    def step():
        """
        FOR TRAINING ONLY
        
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
            loss, global_norm, global_step
        """
        X_batch, y_batch = zip(*batch)    # Unzip batch, each returned element is a tuple of lists

        input_feed = {}

        input_feed[self.X] = X_batch
        input_feed[self.y] = y_batch
        input_feed[self.is_training] = True
        input_feed[self.learning_rate] = self.current_lr

        output_feed = []

        output_feed.append(self.train_op)
        output_feed.append(self.loss)
        output_feed.append(self.global_norm)
        output_feed.append(self.global_step)

        tr, loss, norm, step = self.sess.run(output_feed, input_feed)

        return loss, norm, step

    def optimize():
        # Epoch loop
            # Training loop
                # Update step
                # Print info
            # Validate model
            # Save checkpoint
        pass