
import os
from glob import glob
from pydub import AudioSegment
from pydub.utils import make_chunks
import random
from shutil import copyfile
import tensorflow as tf

import utils

tf.app.flags.DEFINE_string("data_dir", "data/train", "Data directory containing all audio files")
tf.app.flags.DEFINE_integer("clip_duration_ms", 1000, "Expected duration in milliseconds of the wavs.")
FLAGS = tf.app.flags.FLAGS

# Set up the unknown and silence directories
NUM_EXAMPLES_PER_CLASS = 2350


def setup_silence_examples(FLAGS):
    directory = os.path.join(FLAGS.data_dir, "audio/silence")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Find all of the background noise files
    bkgrnd_dir = os.path.join(FLAGS.data_dir, "audio/_background_noise_")
    bkgrnd_glob = os.path.join(bkgrnd_dir, "*.wav")
    print(bkgrnd_glob)
    bkgrnd_files = glob(bkgrnd_glob)
    print("Sampling from these files: ", bkgrnd_files)

    # Loop until we get to NUM_EXAMPLES_PER_CLASS
    counter = 0
    while True:
        random.shuffle(bkgrnd_files)
        for bkgrnd_file in bkgrnd_files:
            # Split into chunks
            wav = AudioSegment.from_file(bkgrnd_file, "wav")
            chunks = make_chunks(wav, FLAGS.clip_duration_ms)
            chunks = chunks[:-1]  # Cut out the last chunk, which isn't 1 sec
            bkgrnd_name = os.path.basename(bkgrnd_file)[:-4]  # Remove '.wav'


            # Export all of the individual chunks as wav files
            for chunk in chunks:
                chunk_name = os.path.join(directory, "{}{}.wav".format(bkgrnd_name, counter))
                print("exporting", chunk_name)
                chunk.export(chunk_name, format="wav")
                counter += 1
                if counter >= NUM_EXAMPLES_PER_CLASS:
                    return


def setup_unknown_examples(FLAGS):
    directory = os.path.join(FLAGS.data_dir, "audio/unknown")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get all "unknown" labels
    labels = set(utils.label_to_num.keys()) - set(utils.small_label_to_num.keys())
    print("'unknown' labels: ", labels)
    label_dirs = [os.path.join(FLAGS.data_dir, "audio", l) for l in labels]

    for i in range(NUM_EXAMPLES_PER_CLASS):
        # Choose random label
        label_dir = random.choice(label_dirs)
        label = os.path.basename(label_dir)

        # Choose random file
        wav_glob = os.path.join(label_dir, "*.wav")
        wav_files = glob(wav_glob)
        wav = random.choice(wav_files)

        # Copy file
        new_name = os.path.join(directory, "{}{}.wav".format(label, i))
        print("copying to", new_name)
        copyfile(wav, new_name)


setup_silence_examples(FLAGS)
setup_unknown_examples(FLAGS)