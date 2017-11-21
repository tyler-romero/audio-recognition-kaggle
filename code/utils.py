import random
import math
import numpy as np
import tensorflow as tf

label_to_num = {
    "bed": 0,
    "bird": 1,
    "cat": 2,
    "dog": 3,
    "down": 4,
    "eight": 5,
    "five": 6,
    "four": 7,
    "go": 8,
    "happy": 9,
    "house": 10,
    "left": 11,
    "marvin": 12,
    "nine": 13,
    "no": 14,
    "off": 15,
    "on": 16,
    "one": 17,
    "right": 18,
    "seven": 19,
    "sheila": 20,
    "six": 21,
    "stop": 22,
    "three": 23,
    "tree": 24,
    "two": 25,
    "up": 26,
    "wow": 27,
    "yes": 28,
    "zero": 29,
    "_background_noise_": 30
}

small_label_to_num = {
    "yes": 0,
    "no": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "on": 6,
    "off": 7,
    "stop": 8,
    "go": 9,
    "silence": 10,
    "unknown": 11
}


def gather_params(FLAGS):
    params = {
        "debug": FLAGS.debug,
        "num_classes": FLAGS.num_classes,
        "sample_rate": FLAGS.sample_rate,
        "epochs": FLAGS.epochs,
        "initial_learning_rate": FLAGS.learning_rate,
        "batch_size": FLAGS.batch_size,
        "model_architecture": FLAGS.model_architecture
    }
    return params


def get_batches(X, y, batch_size):
    dataset = list(zip(X, y))
    random.shuffle(dataset)
    X, y = zip(*dataset)    # Returns a tuple of lists
    num_batches = int(math.ceil(len(X)/float(batch_size)))

    X_batches = []
    y_batches = []
    for i in range(num_batches):
        start_ind = i * batch_size
        end_ind = min(len(dataset), i * batch_size + batch_size)
        X_batches.append(X[start_ind:end_ind])
        y_batches.append(y[start_ind:end_ind])

    return X_batches, y_batches, num_batches


def get_number_of_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def calc_range(l):
    min_val = min(l)
    max_val = max(l)
    return (max_val - min_val) + 1


def get_num_classes(FLAGS):
    if FLAGS.debug:
        return 3
    elif FLAGS.competition_labels:
        return len(small_label_to_num)
    else:
        return len(label_to_num)