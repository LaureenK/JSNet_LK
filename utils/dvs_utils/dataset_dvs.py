# -*- coding: utf-8 -*-
import os
import glob
import random
from multiprocessing import Pool

import numpy as np

NUM_CLASSES = 4
NUM_POINTS = 2**14  ## <-- TODO: this dataset adapter only supprts UP(!) sampling not DOWN sampling

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

            points.append(np.array([x, y, t], dtype=np.float32))
            labels.append(class_label)
            instances.append(instance_label)
    #shuffle data
    npPoints = np.array(points, dtype=np.float32)
    npSeg = np.array(labels, dtype=np.uint8)
    npIns = np.array(instances, dtype=np.uint8)

    npPoints, npSeg, npIns = unison_shuffled_copies(npPoints, npSeg, npIns)
 
    return npPoints, npSeg, npIns

def upscale(points, labels, instances):
    #if len(points) > NUM_POINTS:
        #raise RuntimeError("no matching config...!")

    while len(points) < NUM_POINTS:
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

def create_two_x(points, labels, instances):
    small_points1 = []
    small_labels1 = []
    small_instances1 = []

    small_points2 = []
    small_labels2 = []
    small_instances2 = []

    i = 0
    while i < len(points):
        if(points[i][0] < 320):
            small_points1.append(points[i])
            small_labels1.append(labels[i])
            small_instances1.append(instances[i])
        else:
            small_points2.append(points[i])
            small_labels2.append(labels[i])
            small_instances2.append(instances[i])

        i = i + 1

    small_points3 = []
    small_labels3 = []
    small_instances3 = []

    small_points1, small_labels1, small_instances1 = upscale(small_points1, small_labels1, small_instances1)
    small_points2, small_labels2, small_instances2 = upscale(small_points2, small_labels2, small_instances2)

    small_points3.append(small_points1)
    small_labels3.append(small_labels1)
    small_instances3.append(small_instances1)
    small_points3.append(small_points2)
    small_labels3.append(small_labels2)
    small_instances3.append(small_instances2)

    return small_points3, small_labels3, small_instances3

def create_two_y(points, labels, instances):
    small_points1 = []
    small_labels1 = []
    small_instances1 = []

    small_points2 = []
    small_labels2 = []
    small_instances2 = []

    i = 0
    while i < len(points):
        if(points[i][1] < 384):
            small_points1.append(points[i])
            small_labels1.append(labels[i])
            small_instances1.append(instances[i])
        else:
            small_points2.append(points[i])
            small_labels2.append(labels[i])
            small_instances2.append(instances[i])

        i = i + 1

    small_points3 = []
    small_labels3 = []
    small_instances3 = []

    small_points1, small_labels1, small_instances1 = upscale(small_points1, small_labels1, small_instances1)
    small_points2, small_labels2, small_instances2 = upscale(small_points2, small_labels2, small_instances2)

    small_points3.append(small_points1)
    small_labels3.append(small_labels1)
    small_instances3.append(small_instances1)
    small_points3.append(small_points2)
    small_labels3.append(small_labels2)
    small_instances3.append(small_instances2)

    return small_points3, small_labels3, small_instances3

def downscale(points, labels, instances, x=True):
    small_points = []
    small_labels = []
    small_instances = []

    print("Before downscale X: ", len(points))

    if x == True:
        small_points1, small_labels1, small_instances1 = create_two_x(points, labels, instances)
        print("After downscale X: ", len(small_points1[0]))
        print("After downscale X: ", len(small_points1[1]))

        if len(small_points1[0]) > NUM_POINTS:
            small_points2, small_labels2, small_instances2 = downscale(small_points1[0], small_labels1[0], small_instances1[0], False)
            i = 0
            while i < len(small_points2):
                small_points.append(small_points2[i])
                small_labels.append(small_labels2[i])
                small_instances.append(small_instances2[i])
                i = i +1
        else:
            i = 0
            while i < len(small_points1):
                small_points.append(small_points1[i])
                small_labels.append(small_labels1[i])
                small_instances.append(small_instances1[i])
                i = i +1
    else:
        small_points1, small_labels1, small_instances1 = create_two_y(points, labels, instances)
        print("After downscale Y: ", len(small_points1[0]))
        print("After downscale Y: ", len(small_points1[1]))

        if len(small_points1[0]) > NUM_POINTS:
            small_points2, small_labels2, small_instances2 = downscale(small_points1[0], small_labels1[0], small_instances1[0], True)
            i = 0
            while i < len(small_points2):
                small_points.append(small_points2[i])
                small_labels.append(small_labels2[i])
                small_instances.append(small_instances2[i])
                i = i +1
        else:
            i = 0
            while i < len(small_points1):
                small_points.append(small_points1[i])
                small_labels.append(small_labels1[i])
                small_instances.append(small_instances1[i])
                i = i +1

    return small_points, small_labels, small_instances

class DVSDataset():
    def __init__(self, data_root, input_list_txt = 'none', npoints=16384, split='train', batchsize=24):
        random.seed(1337)  # same result every time

        self.input_list_txt = input_list_txt
        self.split = split
        self.data_root = data_root
        self.batch_count = 0
        self.batchsize = batchsize
        
        

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

        random.shuffle(self.files_to_use)
        self.length = len(self.files_to_use)
        self.batch_num = self.length // batchsize

        # --------------------------------------------------------------------------------------------------------------
        if split not in ['train', 'validation', 'train']:
            raise ValueError("unknown split")

        # parallel csv read...
        pool = Pool(processes=None)
        points, labels, instances = zip(*pool.map(load_and_upscale, self.files_to_use))
        points, labels, instances = self.downscale(list(points), list(labels), list(instances))
        
        self.point_list = np.asarray(points)
        self.semantic_label_list = np.asarray(labels)
        self.instance_label_list = np.asarray(instances)

        #print(len(self.point_list), len(self.semantic_label_list), len(self.instance_label_list))
        print(self.point_list.shape, self.semantic_label_list.shape, self.instance_label_list.shape)
        

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
        #return len(self.point_list)
        return self.length

    def downscale(self, points, labels, instances):
        too_big_points = []
        too_big_labels = []
        too_big_instances = []

        n=0

        while n < len(points):
            if(len(points[n]) > NUM_POINTS):
                too_big_points.append(points.pop(n))
                too_big_labels.append(labels.pop(n))
                too_big_instances.append(instances.pop(n))
            n = n + 1
        print("Count to big: ", len(too_big_points))

        n=0
        while n < len(too_big_points):
            small_points, small_labels, small_instances = downscale(too_big_points[n], too_big_labels[n],too_big_instances[n])
            i=0
            while i < len(small_points):
                points.append(small_points[i])
                labels.append(small_labels[i])
                instances.append(small_instances[i])
                i = i+1

            n = n + 1

        print("length after downscale: ", len(points))
        return tuple(points), tuple(labels), tuple(instances)

    def __getitem__(self, index):
        return self.point_list[index], \
               self.semantic_label_list[index].astype(np.int32), \
               self.labelweights[self.semantic_label_list[0].astype(np.int32)]
               
    def get_batch(self, data_aug=False):

        points = self.point_list[(self.batch_count*self.batchsize):((self.batch_count+1)*self.batchsize)][:][:]
        sem = self.semantic_label_list[(self.batch_count*self.batchsize):((self.batch_count+1)*self.batchsize)][:][:]
        inst = self.instance_label_list[(self.batch_count*self.batchsize):((self.batch_count+1)*self.batchsize)][:][:]

        self.batch_count = self.batch_count + 1
        if(self.batch_count == self.batch_num):
            self.batch_count = 0
            
        return points, sem, inst

    def get_input_list(self):
        input_list = [line.strip() for line in open(self.input_list_txt, 'r')]

        #temp_list = [item.split('/')[-1].strip('.h5').strip('.npy').strip('.csv') for item in input_list]
 
        #cnt_length = len(temp_list)
        #self.length = cnt_length

        input_list = [os.path.join(self.data_root, item) for item in input_list]

        return input_list

    def get_length(self):
        return self.__len__()
# ------------------------------------------------------------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------
    dvsDataset = DVSDataset('data', '/home/klein/neural_networks/jsnet/JSNet_LK/data/train_csv_dvs.txt', split='train', batchsize=8)
    # points, sem, inst = dvsDataset.get_batch()
    # print(points.shape)
    # print(sem.shape)
    # print(inst.shape)
    # points, sem, inst = dvsDataset.get_batch()
    # print(points.shape)
    # print(sem.shape)
    # print(inst.shape)
    # points, sem, inst = dvsDataset.get_batch()
    # print(points.shape)
    # print(sem.shape)
    # print(inst.shape)
