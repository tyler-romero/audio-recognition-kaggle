# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts a trained checkpoint into a frozen model for mobile inference.

Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:

bazel run tensorflow/examples/speech_commands/freeze -- \
--sample_rate=16000 --dct_coefficient_count=40 --window_size_ms=20 \
--window_stride_ms=10 --clip_duration_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

One thing to watch out for is that you need to pass in the same arguments for
`sample_rate` and other command line variables here as you did for the training
script.

The resulting graph has an input for WAV-encoded data named 'wav_data', one for
raw PCM data (as floats in the range -1.0 to 1.0) called 'decoded_sample_data',
and the output is called 'labels_softmax'.

"""
import os
import sys

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util

import utils
import data_utils
import framework
import models

# Loading options
tf.app.flags.DEFINE_string("model_architecture", "specify_a_model", "The name of the model.")
tf.app.flags.DEFINE_string("models_dir", "models", "Directory to save tf model checkpoints in")
tf.app.flags.DEFINE_string("save_name", "model.ckpt", "Name under which to save the model.")
tf.app.flags.DEFINE_string("output_file", "pb/model.pb", "Where to save the frozen graph.")

# Dont mess with these for now:
tf.app.flags.DEFINE_integer("sample_rate", 16000, "Expected sample rate of the wavs.")
tf.app.flags.DEFINE_integer("clip_duration_ms", 1000, "Expected duration in milliseconds of the wavs.")
tf.app.flags.DEFINE_integer("clip_stride_ms", 30, "How often to run recognition. Useful for models with cache.")
tf.app.flags.DEFINE_integer("dct_coefficient_count", 40, "How many bins to use for the MFCC fingerprint.")
tf.app.flags.DEFINE_float("window_size_ms", 30.0, "How long each spectrogram timeslice is.")
tf.app.flags.DEFINE_float("window_stride_ms", 10.0, "How long the stride is between spectrogram timeslices.")
tf.app.flags.DEFINE_float("time_shift_ms", 100.0, "Range to randomly shift the training audio by in time.")

FLAGS = tf.app.flags.FLAGS
FLAGS.restore = True
FLAGS.learning_rate = 0
FLAGS.batch_size = 1
FLAGS.epoch = 0

# TODO: test this on a raspberry pi

# Load correct model and add nodes needed for practical use
def create_inference_graph_and_load_variables(sess, FLAGS):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output the trained model graph.
    """
    model_settings = data_utils.prepare_settings(
        FLAGS.num_classes, FLAGS.sample_rate, FLAGS.clip_duration_ms,
        FLAGS.window_size_ms, FLAGS.window_stride_ms, FLAGS.dct_coefficient_count
    )
    runtime_settings = {'clip_stride_ms': FLAGS.clip_stride_ms}

    wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
    decoded_sample_data = contrib_audio.decode_wav(
        wav_data_placeholder,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_data'
    )
    spectrogram = contrib_audio.audio_spectrogram(
        decoded_sample_data.audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True
    )
    fingerprint_input = contrib_audio.mfcc(
        spectrogram,
        decoded_sample_data.sample_rate,
        dct_coefficient_count=FLAGS.dct_coefficient_count
    )
    fingerprint_frequency_size = model_settings['dct_coefficient_count']
    fingerprint_time_size = model_settings['spectrogram_length']
    reshaped_input = tf.reshape(
        fingerprint_input,
        [-1, fingerprint_time_size, fingerprint_frequency_size, 1],
        name="model_input"
    )

    # Init model and load variables
    model = models.create_model(FLAGS)
    fw = framework.Framework(sess, model, None, FLAGS, input_tensor=reshaped_input)

    # Create an output to use for inference
    logits = tf.nn.softmax(model.get_raw_scores(), name='labels_softmax')


def main(_):
    # Create the model and load its weights.
    FLAGS.output_path = os.path.join(FLAGS.models_dir, FLAGS.output_file)
    FLAGS.num_classes = len(utils.small_label_to_num)

    # Get the trained model
    sess = tf.InteractiveSession()
    create_inference_graph_and_load_variables(sess, FLAGS)

    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax']
    )

    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(FLAGS.output_path),
        os.path.basename(FLAGS.output_path),
        as_text=False
    )
    tf.logging.info('Saved frozen graph to %s', FLAGS.output_path)

    # Write out graph for debugging
    # g = tf.Graph()
    # with g.as_default():
    #     returned_tensors = tf.import_graph_def(
    #         frozen_graph_def,
    #         input_map=None,
    #         return_elements=["labels_softmax"],
    #         name=""
    #     )
    #     tf.summary.FileWriter("logs", g).close()


if __name__ == "__main__":
    tf.app.run()