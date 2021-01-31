from utils import load_image
from image_processing import pre_cnn_image_processing
import numpy as np
import os
from os import listdir
import json

DIGITS_SYMBOLS_MAPPING = "digits-symbols-mapping.json"
DATASET_PATH = "../Kaggle/Kaggle_reducted_dataset/"
WIDTH_DS_ITEM = 28   # Dataset sample width
HEIGHT_DS_ITEM = 28  # Dataset sample height

class DigitsSymbolsDataset:
    def __init__(self):
        self.__targets = []
        self.__samples = []

    def __add_sample(self, sample):
        self.__samples.append(sample)

    def __get_samples(self):
        return self.__samples

    def __add_target(self, target):
        self.__targets.append(target)

    def __get_targets(self):
        return self.__targets

    def load_data(self):
        # Load digits-symbols mapping from categorical to numerical
        with open(DIGITS_SYMBOLS_MAPPING, "r") as dig_sym_mapping_file:
            labels_mapping = json.load(dig_sym_mapping_file)["DIGITS_SYMBOLS_MAPPING"]

        # Load Kaggle Digits and Symbols
        labels = np.array(listdir(DATASET_PATH))
        for label in labels:
            data = np.array(os.listdir(DATASET_PATH + label))
            for sample in data:
                # print("- ", label, " --> ", sample)
                self.__add_target(labels_mapping[label])
                filename = DATASET_PATH + label + "/" + sample
                im_sample = pre_cnn_image_processing(load_image(filename))
                self.__add_sample(im_sample)

        # Join images and targets in a single list for shuffling it
        images_and_labels = list(zip(self.__get_samples(), self.__get_targets()))
        np.random.shuffle(images_and_labels)

        # Split data in training and test
        test_set_percent = 0.1
        train_set_percent_len = int(len(images_and_labels) * (1.0 - test_set_percent))

        train_set = images_and_labels[:train_set_percent_len]
        test_set = images_and_labels[train_set_percent_len:]

        train_set = list(zip(*train_set))
        test_set = list(zip(*test_set))

        x_train = np.array(train_set[0])
        y_train = np.array(train_set[1])
        x_test = np.array(test_set[0])
        y_test = np.array(test_set[1])

        return (x_train, y_train), (x_test, y_test)
