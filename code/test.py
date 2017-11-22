import os
import sys

import numpy as np
import tensorflow as tf
from comet_ml import Experiment

import data_utils
import models
import framework
import utils


tf.app.flags.DEFINE_string("model_architecture", "Baseline", "The name of the model.")
tf.app.flags.DEFINE_string("data_dir", "data/train", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_string("models_dir", "models", "Directory to save tf model checkpoints in")
tf.app.flags.DEFINE_string("save_name", "model.ckpt", "Name under which to save the model.")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")
tf.app.flags.DEFINE_bool("competition_labels", True, "Run only on the ten competiton lables.")

# Dont mess with these for now:
tf.app.flags.DEFINE_integer("sample_rate", 16000, "Expected sample rate of the wavs.")
tf.app.flags.DEFINE_integer("clip_duration_ms", 1000, "Expected duration in milliseconds of the wavs.")
tf.app.flags.DEFINE_integer("clip_stride_ms", 30, "How often to run recognition. Useful for models with cache.")
tf.app.flags.DEFINE_integer("dct_coefficient_count", 40, "How many bins to use for the MFCC fingerprint.")
tf.app.flags.DEFINE_float("window_size_ms", 30.0, "How long each spectrogram timeslice is.")
tf.app.flags.DEFINE_float("window_stride_ms", 10.0, "How long the stride is between spectrogram timeslices.")
tf.app.flags.DEFINE_float("time_shift_ms", 100.0, "Range to randomly shift the training audio by in time.")

# TODO: Allow for different data types ie tf.float16 instead of tf.float32

FLAGS = tf.app.flags.FLAGS
FLAGS.restore = True
FLAGS.learning_rate = 0
FLAGS.batch_size = 1
FLAGS.epochs = 0


def main(_):
    print("Model Architecture: {}".format(FLAGS.model_architecture))

    # Adjust some parameters
    if FLAGS.debug:
        FLAGS.competition_labels = False
        print("RUNNING IN DEBUG MODE")

    FLAGS.num_classes = utils.get_num_classes(FLAGS)

    X_train, y_train = data_utils.load_dataset_tf(FLAGS, mode="train")
    X_val, y_val = data_utils.load_dataset_tf(FLAGS, mode="val")
    X_test = data_utils.load_dataset_tf(FLAGS, mode="test")

    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new, DEFAULT TensorFlow session.
    sess = tf.InteractiveSession()

    model = models.create_model(FLAGS)  
    fw = framework.Framework(sess, model, None, FLAGS)

    num_params = int(utils.get_number_of_params())
    model_size = num_params * 4
    print("\nNumber of trainable parameters: {}".format(num_params))
    print("Model is ~ {} bytes out of max 5000000 bytes\n".format(model_size))

    for X in X_train:
        pred = fw.classify(X_train)


if __name__ == "__main__":
    tf.app.run()