import sys
import os
import cv2
import gist
from tqdm import tqdm
from scipy.spatial import distance as dist
import numpy as np
from utils.config import *
from shutil import copyfile, rmtree
import json


def val_distances(features_vectors, res, fold, distance_algo='eucl'):
    name_folder = fold.split(os.sep)[-1]
    error = 0
    sparse_distance = {'<0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '>0.3': 0}
    if len(features_vectors) > 1:
        for i in range(len(features_vectors)):
            d = 0
            for j in range(len(features_vectors)):
                if i == j:
                    continue
                if distance_algo == 'eucl':
                    temp_dist = dist.euclidean(features_vectors[i], features_vectors[j])
                else:
                    temp_dist = np.linalg.norm(features_vectors[i] - features_vectors[j])

                if temp_dist < 0.1:
                    sparse_distance['<0.1'] += 1
                elif temp_dist < 0.2:
                    sparse_distance['0.1-0.2'] += 1
                elif temp_dist < 0.3:
                    sparse_distance['0.2-0.3'] += 1
                else:
                    sparse_distance['>0.3'] += 1
                d += temp_dist
            error += d / (len(features_vectors)-1)

        errorAVG = error / len(features_vectors)
        res[name_folder] = {}
        res[name_folder]['tot'] = error
        res[name_folder]['AVG'] = errorAVG
        res[name_folder]['sparse_dist'] = sparse_distance
        tot = sparse_distance['<0.1'] + sparse_distance['0.1-0.2'] + \
              sparse_distance['0.2-0.3'] + sparse_distance['>0.3']
        res[name_folder]['sparse_distAVG'] = {'<0.1': sparse_distance['<0.1']/tot,
                                              '0.1-0.2': sparse_distance['0.1-0.2']/tot,
                                              '0.2-0.3': sparse_distance['0.2-0.3'] / tot,
                                              '>0.3': sparse_distance['>0.3']/tot}

    else:
        zero_distance_families.append(name_folder)
        res[name_folder] = {}
        res[name_folder]['tot'] = 0
        res[name_folder]['AVG'] = 0
        res[name_folder]['sparse_dist'] = {}

    return res



# file = sys.argv[1]
# temp_path = sys.argv[2]
nblocks = 4
orientations_per_scale = (8, 8, 4)

# get list of folders with not MIX_ data
class_folders_original = [x[0] for x in os.walk(main_path + 'results/images')
                          if not x[0].split(os.sep)[-1].startswith("MIX_") and not x[0].endswith("images")]

fold = 1
results = {}
results2 = {}
zero_distance_families = []

# calculate distance for family folders
for folder in class_folders_original:
    features_list = []
    heatmap_list = []
    print("FOLDER {} out of {}".format(fold, len(class_folders_original)))
    fold += 1
    for file in tqdm(os.listdir(folder)):
        if not file.startswith('heatmap_'):
            continue
        filepath = folder + os.sep + file
        heatmap = cv2.imread(filepath)
        # skip the file if the heatmap contains only zeros
        if np.all((heatmap == 0)):
            continue

        # img: A numpy array (an instance of numpy.ndarray) which contains an image and whose shape is (height, width, 3).
        # nblocks: Use a grid of nblocks * nblocks cells.
        # orientations_per_scale: Use len(orientations_per_scale) scales and compute orientations_per_scale[i] orientations
        #   for i-th scale.
        features = gist.extract(heatmap, nblocks=nblocks, orientations_per_scale=orientations_per_scale)
        #np.save(temp_path, features)
        # print(features)
        features_list.append(features)
        heatmap_list.append(heatmap)

    results = val_distances(features_list, results, folder)
    #results2 = val_distances(heatmap_list, results2, folder, distance_algo='linalg')

# Generate MIXED folders
print("Generating Mixed Folders")
MIXED = True
if MIXED:
    NUM_HEATMAP = 50
    mix = 2
    while mix <= len(class_folders_original):
        toMix = []
        name = ""
        for ind in range(len(class_folders_original)):
            if class_folders_original[ind].split("/")[-1] in zero_distance_families:
                continue
            toMix.append(ind)
            name += "_" + class_folders_original[ind].split("/")[-1]
            if len(toMix) % mix == 0:
                HEATMAP_PER_FOLDER = int(NUM_HEATMAP / mix)
                if os.path.isdir(main_path + 'results/images/MIX' + name):
                    rmtree(main_path + 'results/images/MIX' + name)
                os.mkdir(main_path + 'results/images/MIX' + name)
                k = 0
                for i in range(len(toMix)):
                    for file in range(HEATMAP_PER_FOLDER):
                        if os.path.isfile(class_folders_original[i] + "/heatmap_" + str(k) + ".png"):
                            copyfile(class_folders_original[i] + "/heatmap_" + str(k) + ".png",
                                     main_path + 'results/images/MIX' + name + "/heatmap_" + str(k) + ".png")
                        elif os.path.isfile(class_folders_original[i] + "/heatmapWRONG_" + str(k) + ".png"):
                            copyfile(class_folders_original[i] + "/heatmapWRONG_" + str(k) + ".png",
                                     main_path + 'results/images/MIX' + name + "/heatmap_W" + str(k) + ".png")
                        k += 1

                toMix = []
                name = ""
        mix += 1

class_folders_MIX = [x[0] for x in os.walk(main_path + 'results/images')
                          if not x[0].endswith("images") and "MIX_" in x[0]]
fold = 1

# calculate distance for MIX folders
for folder in class_folders_MIX:
    features_list = []
    heatmap_list = []
    print("FOLDER {} out of {}".format(fold, len(class_folders_MIX)))
    fold += 1
    for file in tqdm(os.listdir(folder)):
        if not file.startswith('heatmap_'):
            continue
        filepath = folder + os.sep + file
        heatmap = cv2.imread(filepath)
        # skip the file if the heatmap contains only zeros
        if np.all((heatmap == 0)):
            continue

        # img: A numpy array (an instance of numpy.ndarray) which contains an image and whose shape is (height, width, 3).
        # nblocks: Use a grid of nblocks * nblocks cells.
        # orientations_per_scale: Use len(orientations_per_scale) scales and compute orientations_per_scale[i] orientations
        #   for i-th scale.
        features = gist.extract(heatmap, nblocks=nblocks, orientations_per_scale=orientations_per_scale)
        #np.save(temp_path, features)
        # print(features)
        features_list.append(features)
        heatmap_list.append(heatmap)

    results = val_distances(features_list, results, folder)
    #results2 = val_distances(heatmap_list, results2, folder, distance_algo='linalg')

print(results)

# Print results on file, with an incremental 'j' to not overwrite previous results
j = 0
while os.path.isfile(main_path + 'results/images/results{}.json'.format(j)):
    j += 1

with open(main_path + 'results/images/results{}.json'.format(j), 'w') as outfile:
    json.dump(results, outfile, indent=4)

