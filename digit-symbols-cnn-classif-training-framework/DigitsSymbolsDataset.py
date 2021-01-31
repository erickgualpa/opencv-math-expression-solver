# TODO: HANDLE import load_image, pre_cnn_image_processing
from os import listdir
import numpy as np
import os
import json

from constants import DIGITS_SYMBOLS_MAPPING, DATASET_PATH


class DigitsSymbolsDataset:
    def __init__(self):

        self.__targets = []
        self.__samples = []
        self.__train_set = []
        self.__test_set = []

    def __add_sample(self, sample):
        self.__samples.append(sample)

    def __get_samples(self):
        return self.__samples

    def __add_target(self, target):
        self.__targets.append(target)

    def __get_targets(self):
        return self.__targets

    def __load_digits_symbols_mapping(self):

        # Load digits-symbols mapping from categorical to numerical
        labels_mapping = None
        with open(DIGITS_SYMBOLS_MAPPING, "r") as dig_sym_mapping_file:
            labels_mapping = json.load(dig_sym_mapping_file)["DIGITS_SYMBOLS_MAPPING"]
        return labels_mapping

    def __load_kaggle_digits_and_symbols(self, labels_mapping):

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

    def __join_images_and_targets_in_single_list(self):

        # Join images and targets in a single list for shuffling it
        images_and_labels = list(zip(self.__get_samples(), self.__get_targets()))
        np.random.shuffle(images_and_labels)
        return images_and_labels

    def __split_data_in_training_in_test(self, images_and_labels):

        # Split data in training and test
        test_set_percent = 0.1
        train_set_percent_len = int(len(images_and_labels) * (1.0 - test_set_percent))

        self.__train_set = images_and_labels[:train_set_percent_len]
        self.__test_set = images_and_labels[train_set_percent_len:]

        self.__train_set = list(zip(*self.__train_set))
        self.__test_set = list(zip(*self.__test_set))

        x_train = np.array(self.__train_set[0])
        y_train = np.array(self.__train_set[1])
        x_test = np.array(self.__test_set[0])
        y_test = np.array(self.__test_set[1])

        return (x_train, y_train), (x_test, y_test)

    def load_data(self):

        labels_mapping = self.__load_digits_symbols_mapping()
        if None != labels_mapping:
            self.__load_kaggle_digits_and_symbols(labels_mapping)
            images_and_labels = self.__join_images_and_targets_in_single_list()
            return self.__split_data_in_training_in_test(images_and_labels)

        return None

