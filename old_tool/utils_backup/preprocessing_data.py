import os
import re
import random
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
import pathlib
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
from old_tool.utils_backup.config import *
from old_tool.utils_backup.training_utils import balanced_split_train_test
from tensorflow import data as tf_data


def extract_data_paths(arguments):
    dataset_training = ['{}/{}'.format(main_path + arguments.dataset, name)
                        for name in os.listdir(main_path + arguments.dataset)]

    # Simple but imperfect way to check if data match my dataset format
    my_data_regex = re.compile(r'/\w{5,11}_variety\d_\w+.apk.png$')
    for f in dataset_training:
        if my_data_regex.search(f) is None:
            print("DATA '{}' does not match my format, needs refactoring, exiting...".format(f))
            exit()

    return dataset_training


def load_and_preprocess(arguments):
    dataset_paths = extract_data_paths(arguments)

    print("PRE-PROCESSING DATA")
    total_data, total_labels, info_labels, class_names = preprocessing_data(arguments, dataset_paths)
    # Normalize Data
    total_data = total_data / 255.0

    # Print info on dataset
    print("Dataset size is:", total_data.shape)
    print("The classes ({}) are: {}".format(len(info_labels), class_names))

    # Lets create the augmentation configuration, this helps prevent overfitting
    # train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
    #                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # train_datagen = ImageDataGenerator(rescale=1. / 255)
    # validation and test only rescaling of images
    # val_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # data_generators = [train_datagen, val_datagen, test_datagen]

    # Split the dataset into total_training and test set
    if arguments.rand_split:
        total_train_data, test_data, total_train_labels, test_labels = train_test_split(total_data, total_labels,
                                                                                        test_size=arguments.testing,
                                                                                        random_state=42, shuffle=True)
    else:
        ind_total_train, ind_test = balanced_split_train_test(total_labels, split=arguments.testing)
        total_train_data, total_train_labels = total_data[ind_total_train], total_labels[ind_total_train]
        test_data, test_labels = total_data[ind_test], total_labels[ind_test]

    del dataset_paths, total_data, total_labels

    return info_labels, class_names, \
           total_train_data, test_data, total_train_labels, test_labels


def load_and_preprocess_OPTIMIZED(arguments):
    dataset_paths = extract_data_paths(arguments)

    print("PRE-PROCESSING DATA")
    total_data, total_labels, info_labels, class_names = preprocessing_data(arguments, dataset_paths)
    # Normalize Data
    total_data = total_data / 255.0

    # Print info on dataset
    print("Dataset size is:", total_data.shape)
    print("The classes ({}) are: {}".format(len(info_labels), class_names))

    # Lets create the augmentation configuration, this helps prevent overfitting
    # train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2,
    #                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # train_datagen = ImageDataGenerator(rescale=1. / 255)
    # validation and test only rescaling of images
    # val_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # data_generators = [train_datagen, val_datagen, test_datagen]

    # Split the dataset into total_training and test set
    if arguments.rand_split:
        total_train_data, test_data, total_train_labels, test_labels = train_test_split(total_data, total_labels,
                                                                                        test_size=arguments.testing,
                                                                                        random_state=42, shuffle=True)
    else:
        ind_total_train, ind_test = balanced_split_train_test(total_labels, split=arguments.testing)
        total_train_data, total_train_labels = total_data[ind_total_train], total_labels[ind_total_train]
        test_data, test_labels = total_data[ind_test], total_labels[ind_test]

    del dataset_paths, total_data, total_labels

    temp_storage_tot_train = main_path + 'temp/total_train.data'
    temp_storage_test = main_path + 'temp/test.data'

    # delete previous temp data and store for this execution
    shutil.rmtree(main_path + 'temp/*.data', ignore_errors=True)
    with open(temp_storage_tot_train, 'wb') as filehandle:
        store_data = {"data": total_train_data, "labels": total_train_labels}
        pickle.dump(store_data, filehandle)
    with open(temp_storage_test, 'wb') as filehandle:
        store_data = {"data": test_data, "labels": test_labels}
        pickle.dump(store_data, filehandle)

    train_shape = total_train_data.shape
    test_shape = test_data.shape
    del total_train_data, total_train_labels, test_data, test_labels

    return info_labels, class_names, train_shape, temp_storage_tot_train, test_shape, temp_storage_test


def preprocessing_data(arguments, images_paths):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X_data = []  # images
    Y_data = []  # labels
    info_per_labels = {}

    balance_string = "y" if arguments.balance else "n"

    storing_filename = main_path + 'preprocessed_dataset/b{}_s{}_{}x{}x{}.data' \
        .format(balance_string, len(images_paths), arguments.image_size, arguments.image_size, arguments.channels)

    if os.path.isfile(storing_filename):
        print("LOADING DATA from '{}'".format(storing_filename))
        with open(storing_filename, 'rb') as filehandle:
            stored_data = pickle.load(filehandle)
            X_data = stored_data["X_data"]
            Y_data = stored_data["Y_data"]
            info_per_labels = stored_data["info_per_labels"]
            class_names = stored_data["class_names"]

    else:
        index_label = 0
        match_label_encode = {}
        # Visit all the images_path and extract the labels
        for image_path in images_paths:
            # Look for the label
            temp_label = image_path.split("/")[-1].split("_")[0]
            # Update the struct with the label information
            if temp_label not in match_label_encode:
                match_label_encode[temp_label] = {"encode": index_label, "paths": [image_path]}
                index_label += 1
            else:
                match_label_encode[temp_label]["paths"].append(image_path)

        if arguments.balance:
            min_size = None
            for label in match_label_encode:
                if min_size is None or min_size > len(match_label_encode[label]["paths"]):
                    min_size = len(match_label_encode[label]["paths"])
            for label in match_label_encode:
                match_label_encode[label]["paths"] = match_label_encode[label]["paths"][:min_size]

        # Initializing 'class_name' list
        class_names = []
        for x in range(len(match_label_encode)):
            class_names.append("__X__")

        for label in tqdm(match_label_encode):
            info_per_labels[label] = {"encode": match_label_encode[label]["encode"],
                                      "size": len(match_label_encode[label]["paths"])}
            # Set class name at right index in class_names
            class_names[match_label_encode[label]["encode"]] = label
            # to shuffle the different varieties (that are not took into account in the analysis)
            random.shuffle(match_label_encode[label]["paths"])
            for image_path in match_label_encode[label]["paths"]:
                temp_image = None
                # Read the image
                if arguments.channels == 1:
                    temp_image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
                                            (arguments.image_size, arguments.image_size),
                                            interpolation=cv2.INTER_CUBIC)
                    # It is necessary to keep the grayscale channel
                    temp_image = temp_image[..., None]
                elif arguments.channels == 3:
                    temp_image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR),
                                            (arguments.image_size, arguments.image_size),
                                            interpolation=cv2.INTER_CUBIC)
                else:
                    print("requesting '{}' channels, error...".format(arguments.channels))
                    exit()

                # Append the image and label
                X_data.append(temp_image)
                Y_data.append(match_label_encode[label]["encode"])

        print("STORING DATA to '{}'".format(storing_filename))
        with open(storing_filename, 'wb') as filehandle:
            store_data = {"X_data": X_data, "Y_data": Y_data, "info_per_labels": info_per_labels,
                          "class_names": class_names}
            pickle.dump(store_data, filehandle)

    #print(info_per_labels)
    #sns.countplot(Y_data)
    #plt.title('Labels for Malware Family')
    #plt.show()

    # one-hot encoding
    Y_data = to_categorical(Y_data)

    # Convert list to numpy array
    x_data = np.array(X_data)
    y_data = np.array(Y_data)

    del X_data, Y_data

    return x_data, y_data, info_per_labels, class_names


def load_from_temp_file(temp_file):
    with open(temp_file, 'rb') as filehandle:
        stored_data = pickle.load(filehandle)

    return stored_data['data'], stored_data['labels']


def get_info_dataset(dataset_path):
    # TODO: Implemnents some checks to verify edits to the dataset from last pickle.dump(data)
    storing_data_path = dataset_path + "/info.txt"

    if os.path.isfile(storing_data_path):
        with open(storing_data_path, 'rb') as filehandle:
            data = pickle.load(filehandle)
            class_info = data['class_info']
            ds_info = data['ds_info']

    else:

        # Create dataset filepaths
        train_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/training/train")
                       for file in f if ".png" in file or ".jpg" in file]
        val_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/training/val")
                     for file in f if ".png" in file or ".jpg" in file]
        final_training_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/training")
                                for file in f if ".png" in file or ".jpg" in file]
        test_paths = [os.path.join(r, file) for r, d, f in os.walk(dataset_path + "/test")
                      for file in f if ".png" in file or ".jpg" in file]

        ds_info = {'train_paths': train_paths, 'val_paths': val_paths, 'test_paths': test_paths,
                   'final_training_paths': final_training_paths}

        class_names = np.array([item.name for item in pathlib.Path(dataset_path + "/training/train").glob('*')])
        nclasses = len(class_names)
        class_info = {"class_names": class_names, "n_classes": nclasses}

        # GENERAL STATS
        size_train = len(train_paths)
        size_val = len(val_paths)
        size_test = len(test_paths)

        class_info.update({"train_size": size_train, "val_size": size_val, "test_size": size_test, 'info': {}})

        for name in class_names:
            size_trainf = sum([len(files) for r, d, files in os.walk(dataset_path + "/training/train/{}".format(name))])
            size_valf = sum([len(files) for r, d, files in os.walk(dataset_path + "/training/val/{}".format(name))])
            size_testf = sum([len(files) for r, d, files in os.walk(dataset_path + "/test/{}".format(name))])
            class_info['info']["{}".format(name)] = {}
            class_info['info']["{}".format(name)]['TRAIN'] = size_trainf
            class_info['info']["{}".format(name)]['VAL'] = size_valf
            class_info['info']["{}".format(name)]['TEST'] = size_testf
            class_info['info']["{}".format(name)]['TOT'] = size_testf + size_valf + size_trainf

        with open(storing_data_path, 'wb') as filehandle:
            data = {'ds_info': ds_info, 'class_info': class_info}
            pickle.dump(data, filehandle)

    return class_info, ds_info

