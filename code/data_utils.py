import os
from scipy.io import wavfile
from glob import glob


# Load a single wave file, return a numpy array
def load_wav(file_name):
    a = wavfile.read(file_name)
    numpy.array(a[1], dtype=float)


def load_dataset(data_dir="../data/train", mode="train"):
    # Get file paths
    val_list_file = os.path.join(data_dir, "validation_list.txt")
    test_list_file = os.path.join(data_dir, "test_list.txt")
    audio_dir = os.path.join(data_dir, "audio/*/")

    # Load metadata
    val_list = [line.strip() for line in open(val_list_file, 'r')]
    test_list = [line.strip() for line in open(test_list_file, 'r')]
    dir_list = glob(audio_dir)

    # Begin loading data
    if mode == "train":
        for label in dir_list:
            print(label)
