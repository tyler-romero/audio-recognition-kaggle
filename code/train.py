import os
import sys

import numpy as np
import tensorflow as tf
from comet_ml import Experiment

import data_utils
import models
import framework
import utils


tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Number of examples per batch.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_string("model_architecture", "Baseline", "The name of the model.")
tf.app.flags.DEFINE_integer("sample_rate", 16000, "Expected sample rate of the wavs.")
tf.app.flags.DEFINE_string("data_dir", "data/train", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_float("time_shift_ms", 100.0, "Range to randomly shift the training audio by in time.")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    print("Is this debug mode: {}".format(FLAGS.debug))
    X_train, y_train, num_classes = data_utils.load_dataset(FLAGS, mode="train")
    X_val, y_val, _ = data_utils.load_dataset(FLAGS, mode="val")
    FLAGS.num_classes = num_classes

    # comet_ml experiment logging (https://www.comet.ml/)
    experiment = Experiment(api_key="J55UNlgtffTDmziKUlszSMW2w", log_code=False)
    experiment.log_multiple_params(utils.gather_params(FLAGS))
    experiment.set_num_of_epocs(FLAGS.epochs)
    experiment.log_dataset_hash(X_train)

    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new, DEFAULT TensorFlow session.
    sess = tf.InteractiveSession()

    model = models.create_model(FLAGS)
    fw = framework.Framework(sess, model, experiment, FLAGS)

    fw.optimize(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    tf.app.run()