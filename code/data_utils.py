import os
from glob import glob, iglob
import numpy as np
from tqdm import tqdm

from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from utils import *

# TODO: balance classes
# TODO: add some silence / unknown labels to validation set


def get_index_from_label(label, FLAGS):
    if FLAGS.small_label_set:
        if label in small_label_to_num:
            index = small_label_to_num[label]
        else:
            index = small_label_to_num["unknown"]
    else:
        index = label_to_num[label]
    return index


# TODO: Data augmentation
#   * Add background noise
#   * Time shifts
# TODO: Include the fake test set? 
def load_dataset_tf(FLAGS, mode="train"):
    if mode not in ["train", "val"]:
        raise Exception("Unrecognized mode")

    with tf.Session() as sess:
        # Set up the tf audio loading graph
        loader = AudioLoader(FLAGS)

        # Get file paths
        val_list_file = os.path.join(FLAGS.data_dir, "validation_list.txt")
        test_list_file = os.path.join(FLAGS.data_dir, "testing_list.txt")
        audio_glob = os.path.join(FLAGS.data_dir, "audio/*/")

        # Load helper data
        val_list = [line.strip() for line in open(val_list_file, 'r')]
        test_list = [line.strip() for line in open(test_list_file, 'r')]
        dir_list = glob(audio_glob)
        dir_list.remove('data/train/audio/unknown/')
        dir_list.remove('data/train/audio/_background_noise_/')

        # In debug mode, restrict to three classes
        if FLAGS.debug:
            dir_list = dir_list[:3]

        # Iterate over each class directory
        X, y = [], []
        for label_dir in dir_list:
            file_list = iglob(os.path.join(label_dir, "*.wav"))
            short_file_list = ['/'.join(f.split('/')[3:]) for f in file_list]
            label = label_dir.split('/')[-2]

            if mode == "val":  # Load only the val files
                files_to_load = set(short_file_list).intersection(val_list)
            else:  # Load all except the val files  # TODO: Should I ignore the test data?
                files_to_load = set(short_file_list).difference(val_list)

            if FLAGS.small_label_set and label not in small_label_to_num:
                    continue
            else:
                print("Loading {}\t{}".format(label_dir, label))

            # Iterate over the files in each directory
            for f_short in files_to_load:
                f = os.path.join(FLAGS.data_dir, "audio", f_short)

                # Load and preprocess data
                mcff = loader.get_mcff(sess, f)
                
                index = get_index_from_label(label, FLAGS)
                
                # Append data/label to list
                X.append(mcff)
                y.append(index)

        # TODO: Add silence

        print("X: ", len(X), "X shape: ", X[0].shape)
        print("y: ", len(y))
        return X, y


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/
def prepare_settings(
        label_count, sample_rate, clip_duration_ms,
        window_size_ms, window_stride_ms, dct_coefficient_count):
    """Calculates common settings needed for all models.
    Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.
    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }
    

# This code adapted from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py
class AudioLoader():
    def __init__(self, FLAGS):
        model_settings = prepare_settings(
            FLAGS.num_classes, FLAGS.sample_rate,
            FLAGS.clip_duration_ms, FLAGS.window_size_ms,
            FLAGS.window_stride_ms, FLAGS.dct_coefficient_count
        )
        runtime_settings = {'clip_stride_ms': FLAGS.clip_stride_ms}

        # Perform preprocessing
        self.wav_data_placeholder = tf.placeholder(tf.string, [], name='wav')
        wav_data = io_ops.read_file(self.wav_data_placeholder)
        decoded_sample_data = contrib_audio.decode_wav(
            wav_data,
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

        # Add channel dimension
        self.reshaped_input = tf.reshape(
            fingerprint_input,
            [fingerprint_time_size, fingerprint_frequency_size, 1]
        )

    def get_mcff(self, sess, wav_file):
        input_feed = {}
        input_feed[self.wav_data_placeholder] = wav_file
        output_feed = [self.reshaped_input]
        mcff = sess.run(output_feed, input_feed)[0]
        return mcff
