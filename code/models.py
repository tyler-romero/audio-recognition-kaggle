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


class Baseline(Model):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def forward():

