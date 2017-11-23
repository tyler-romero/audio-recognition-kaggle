import os
import sys

import numpy as np
import tensorflow as tf
from comet_ml import Experiment
from glob import glob

import data_utils
import models
import framework
import utils


tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Number of examples per batch.")
tf.app.flags.DEFINE_integer("epochs", 30, "Number of epochs to train.")
tf.app.flags.DEFINE_string("model_architecture", "Baseline", "The name of the model.")
tf.app.flags.DEFINE_string("data_dir", "data/train", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_string("models_dir", "models", "Directory to save tf model checkpoints in")
tf.app.flags.DEFINE_string("save_name", "model.ckpt", "Name under which to save the model.")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")
tf.app.flags.DEFINE_bool("competition_labels", True, "Run only on the ten competiton lables.")
tf.app.flags.DEFINE_bool("restore", False, "Restore model from checkpoint.")

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


def main(_):
    print("Model Architecture: {}".format(FLAGS.model_architecture))

    # Adjust some parameters
    if FLAGS.debug:
        FLAGS.competition_labels = False
        print("RUNNING IN DEBUG MODE")

    FLAGS.num_classes = utils.get_num_classes(FLAGS)

    X_train, y_train = data_utils.load_dataset_tf(FLAGS, mode="train")
    X_val, y_val = data_utils.load_dataset_tf(FLAGS, mode="val")

    # comet_ml experiment logging (https://www.comet.ml/)
    experiment = Experiment(
        api_key="J55UNlgtffTDmziKUlszSMW2w", log_code=False
    )
    experiment.log_multiple_params(utils.gather_params(FLAGS))
    experiment.set_num_of_epocs(FLAGS.epochs)
    experiment.log_dataset_hash(X_train)

    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new, DEFAULT TensorFlow session.
    sess = tf.InteractiveSession()

    utils.set_seeds()  # Get deterministic behavior?

    model = models.create_model(FLAGS)  
    fw = framework.Framework(sess, model, experiment, FLAGS)

    num_params = int(utils.get_number_of_params())
    model_size = num_params * 4
    print("\nNumber of trainable parameters: {}".format(num_params))
    print("Model is ~ {} bytes out of max 5000000 bytes\n".format(model_size))
    experiment.log_parameter("num_params", num_params)
    experiment.log_parameter("approx_model_size", model_size)
    
    fw.optimize(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    tf.app.run()