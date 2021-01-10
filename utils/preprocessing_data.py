import os
import numpy as np
import pickle
import pathlib


def get_info_dataset(dataset_path, update=False):
    # TODO: Implements some checks to verify edits to the dataset from last pickle.dump(data)
    storing_data_path = dataset_path + "/info.txt"

    if update and os.path.exists(dataset_path + "/info.txt"):
        os.remove(dataset_path + "/info.txt")

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

        ds_info = {'ds_type': 'images', 'train_paths': train_paths, 'val_paths': val_paths, 'test_paths': test_paths,
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

