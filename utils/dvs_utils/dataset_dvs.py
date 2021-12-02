# -*- coding: utf-8 -*-
import os
import glob
import random
from multiprocessing import Pool

import numpy as np

NUM_CLASSES = 4
NUM_POINTS = 2**16  ## <-- TODO: this dataset adapter only supprts UP(!) sampling not DOWN sampling

DATASET_TRAIN_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TrainFiles/"
DATASET_VALIDATION_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/ValidationFiles/"
DATASET_TEST_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TestFiles/"

CLASS_COLORS = {
    0: (255, 116, 0),    # ClassLabel::PERSON
    1: (92, 255, 0),     # ClassLabel::DOG
    2: (0, 128, 128),    # ClassLabel::BICYCLE
    3: (255, 153, 204),  # ClassLabel::SPORTSBALL
}

CLASS_MAPPING = {
    1: 0,  # original PERSON(1) --> 0
    2: 1,  # original DOG(2) --> 1
    5: 2,  # original BICYCLE(5) --> 2
    6: 3,  # original SPORTSBALL(6) --> 3
}

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def load_ascii_cloud(fname):
    points = []
    labels = []
    instances = []
    newdata = np.zeros((1,5))

    with open(fname, 'r') as fd:
        for line in fd.readlines():
            if "//" in line:
                continue

            x, y, t, pol, class_label, instance_label = line.strip().split(' ')
            x, y, t, pol, class_label, instance_label = float(x), float(y), float(t), int(pol), int(class_label), int(instance_label)

            class_label = CLASS_MAPPING[class_label]
            if class_label not in range(NUM_CLASSES):
                raise ValueError("unknown label!")

            newdata = np.append(newdata,[[x,y,t,class_label,instance_label]], axis=0)
            points.append(np.array([x, y, t], dtype=np.float32))
            labels.append(class_label)
            instances.append(instance_label)
    #shuffle data
    newdata = np.delete(newdata, (0), axis=0)
    print(newdata.shape)
    print(len(points))

    np.array(np.array([]))

    np.random.shuffle(newdata)

    npPoints = np.array(points, dtype=np.float32)
    npSeg = np.array(labels, dtype=np.uint8)
    npIns = np.array(instances, dtype=np.uint8)

    print(npPoints)
    print(npSeg)
    print(npIns)

    npPoints, npSeg, npIns = unison_shuffled_copies(npPoints, npSeg, npIns)

    print(npPoints)
    print(npSeg)
    print(npIns)
 
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.uint8), np.array(instances, dtype=np.uint8)

def upscale(points, labels, instances):
    if len(points) > NUM_POINTS:
        raise RuntimeError("no matching config...!")

    while len(points) != NUM_POINTS:
        copy_index = random.randint(0, len(points)-1)

        points = np.vstack((points, points[copy_index]))
        labels = np.append(labels, labels[copy_index])
        instances = np.append(instances, instances[copy_index])

    return points, labels, instances

def load_and_upscale(path):
    # read all points
    points, labels, instances = load_ascii_cloud(path)

    # just copy some random points... if needed
    points, labels, instances = upscale(points, labels, instances)

    return points, labels, instances


class DVSDataset():
    def __init__(self, data_root, input_list_txt = 'none', npoints=65536, split='train'):
        random.seed(1337)  # same result every time

        self.input_list_txt = input_list_txt
        self.split = split
        self.data_root = data_root

        if npoints != NUM_POINTS:
            raise ValueError("npoints != NUM_POINTS")

        if(input_list_txt == 'none'):
            if(split == 'train'):
                self.files_to_use = glob.glob(os.path.join(DATASET_TRAIN_DIR, split, "*.csv"))
            elif(split == 'validation'): 
                self.files_to_use = glob.glob(os.path.join(DATASET_VALIDATION_DIR, split, "*.csv"))
            elif(split == 'test'):
                self.files_to_use = glob.glob(os.path.join(DATASET_TEST_DIR, split, "*.csv"))
        else:
            self.input_list_txt = input_list_txt
            self.files_to_use = self.get_input_list()

        #random.shuffle(self.files_to_use)

        # --------------------------------------------------------------------------------------------------------------
        if split not in ['train', 'validation', 'train']:
            raise ValueError("unknown split")

        # parallel csv read...
        pool = Pool(processes=None)
        points, labels, instances = zip(*pool.map(load_and_upscale, self.files_to_use))
        self.point_list = points
        self.semantic_label_list = labels
        self.instance_label_list = instances

        print(len(self.point_list), len(self.semantic_label_list), len(self.instance_label_list))
        

        # labelweights
        # TODO: does [e.g. JSnet]-implementation provide some kind of handling of class-imbalances?
        #??
        #if split == 'train':
        #    labelweights = np.zeros(NUM_CLASSES)
        #    for seg in self.semantic_label_list:
        #        tmp, _ = np.histogram(seg, range(NUM_CLASSES + 1))
        #        labelweights += tmp
        #    labelweights = labelweights.astype(np.float32)
        #    labelweights = labelweights / np.sum(labelweights)
         #   self.labelweights = 1 / np.log(1.2 + labelweights)
        #elif split == 'validation':
        #    self.labelweights = np.ones(NUM_CLASSES)

    def __len__(self):
        return len(self.point_list)

    def __getitem__(self, index):
        return self.point_list[index], \
               self.semantic_label_list[index].astype(np.int32), \
               self.labelweights[self.semantic_label_list[0].astype(np.int32)]
               
    def get_batch(self, data_aug=False):
        #return all data (batch size = 8000)

        return self.point_list, self.semantic_label_list, self.instance_label_list

    def get_input_list(self):
        input_list = [line.strip() for line in open(self.input_list_txt, 'r')]

        #temp_list = [item.split('/')[-1].strip('.h5').strip('.npy').strip('.csv') for item in input_list]
 
        #cnt_length = len(temp_list)
        #self.length = cnt_length

        input_list = [os.path.join(self.data_root, item) for item in input_list]

        return input_list

# ------------------------------------------------------------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------
    dvsDataset = DVSDataset('data', '/home/klein/neural_networks/jsnet/JSNet_LK/data/train_csv_dvs.txt', npoints=65536, split='train')

