import os
import sys
import time
from glob import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import utils

tf.app.flags.DEFINE_string("pb", "models/pb/model.pb", "")
tf.app.flags.DEFINE_string("output_file", "answers.csv", "File to write the output predictions to.")
tf.app.flags.DEFINE_string("data_dir", "data/test/audio", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_bool("small_label_set", True, "Run only on the ten competition lables.")

FLAGS = tf.app.flags.FLAGS


# NOTE: This takes about an hour to run on a laptop
def main(_):
    g = tf.Graph()
    with g.as_default():
        graph_def = tf.GraphDef()
        with open(FLAGS.pb, "rb") as f:
            graph_def.ParseFromString(f.read())

        input_wav_placeholder = tf.placeholder(
            tf.string, [], name="input_wav"
        )
        wav_data = io_ops.read_file(input_wav_placeholder)
        returned_tensors = tf.import_graph_def(
            graph_def,
            input_map={"wav_data": wav_data},
            return_elements=["labels_softmax:0"],
            name=""
        )
        labels_softmax = returned_tensors[0]
        prediction = tf.argmax(labels_softmax, axis=1)
        
        print("Input: ", input_wav_placeholder)
        print("Output: ", prediction)

        tensor_names = [n.name for n in g.as_graph_def().node]
        print(tensor_names)

    # Write graph out to make sure it makes sense
    # tf.summary.FileWriter("logs", g).close()

    with tf.Session(graph=g) as sess:
        with open(FLAGS.output_file, 'w') as f:
            print("fname,label", file=f)
            X_test = glob(os.path.join(FLAGS.data_dir, "*.wav"))
            print("Number of test files: {}\n".format(len(X_test)))

            start = time.time()
            for wav_file_path in tqdm(X_test):
                fname = os.path.basename(wav_file_path)

                input_feed = {input_wav_placeholder: wav_file_path}
                pred = int(prediction.eval(input_feed, sess))

                if FLAGS.small_label_set:
                    label = utils.small_num_to_label[pred]
                else:
                    label = utils.num_to_label[pred]
                line = "{},{}".format(fname, label)
                print(line, file=f)
            end = time.time()
            print("Avg. time for a single example: {}".format((end-start)/len(X_test)))


if __name__ == "__main__":
    tf.app.run()