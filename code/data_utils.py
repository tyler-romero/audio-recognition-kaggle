import os
from scipy.io import wavfile
from glob import glob
import numpy as np
from tqdm import tqdm

from utils import label_to_num, WAV_SIZE


# Load a single wave file, return a numpy array
def load_wav(file_name):
    _, wav = wavfile.read(file_name)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav


# Load the dataset, return a 2D Numpy array X: (num_examples, WAV_SIZE)
# and a list of class labels y: (num_examples)
def load_dataset(data_dir="../data/train", mode="train"):
    print("loading dataset")

    # Get file paths
    val_list_file = os.path.join(data_dir, "validation_list.txt")
    test_list_file = os.path.join(data_dir, "testing_list.txt")
    audio_dir = os.path.join(data_dir, "audio/*/")

    # Load metadata
    val_list = [line.strip() for line in open(val_list_file, 'r')]
    test_list = [line.strip() for line in open(test_list_file, 'r')]
    dir_list = glob(audio_dir)

    # Begin loading data
    X = []
    excluded = 0
    if mode == "train":
        y = []

        # Iterate over each class directory
        for label_dir in tqdm(dir_list):
            file_list = glob(os.path.join(label_dir, "*.wav"))
            label = label_dir.split('/')[-2]

            # Iterate over the files in each directory
            for file_path in file_list:
                file_path_short = '/'.join(file_path.split('/')[3:])

                # Exclued test and val files
                if file_path_short in val_list or file_path_short in test_list:
                    continue

                # Impoert wav as numpy array
                wav = load_wav(file_path)

                # For now skip all files that arent 1 sec
                if len(wav) != WAV_SIZE:
                    excluded += 1
                    continue

                X.append(wav)
                y.append(label_to_num[label])
        
        print("examples excluded: {}".format(excluded))
        X = np.asarray(X)
        y = np.asarray(y)
        print("X", X.shape)
        print("y", y.shape)
        return X, y

    elif mode == "val":
        y = []

        # Iterate over each class directory
        for label_dir in tqdm(dir_list):
            file_list = glob(os.path.join(label_dir, "*.wav"))
            label = label_dir.split('/')[-2]

            # Iterate over the files in each directory
            for file_path in file_list:
                file_path_short = '/'.join(file_path.split('/')[3:])

                # Only include val files
                if file_path_short not in val_list:
                    continue

                # Impoert wav as numpy array
                wav = load_wav(file_path)

                # For now skip all files that arent 1 sec
                if len(wav) != WAV_SIZE:
                    excluded += 1
                    continue

                X.append(wav)
                y.append(label_to_num[label])
        
        print("examples excluded: {}".format(excluded))
        X = np.asarray(X)
        y = np.asarray(y)
        print("X", X.shape)
        print("y", y.shape)
        return X, y

    elif mode == "test":
        # Iterate over each class directory
        for label_dir in tqdm(dir_list):
            file_list = glob(os.path.join(label_dir, "*.wav"))
            label = label_dir.split('/')[-2]

            # Iterate over the files in each directory
            for file_path in file_list:
                file_path_short = '/'.join(file_path.split('/')[3:])

                # Only include val files
                if file_path_short not in test_list:
                    continue

                # Impoert wav as numpy array
                wav = load_wav(file_path)

                # For now skip all files that arent 1 sec
                if len(wav) != WAV_SIZE:
                    excluded += 1
                    continue

                X.append(wav)
                y.append(label_to_num[label])
        
        print("examples excluded: {}".format(excluded))
        X = np.asarray(X)
        print("X", X.shape)
        return X

    else:
        raise Exception("Invalid mode")