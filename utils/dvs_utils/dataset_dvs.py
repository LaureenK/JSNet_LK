# -*- coding: utf-8 -*-
import os
import glob
import random
import numpy as np
from multiprocessing import Pool

from dvs_utils.prepareData import prepareData

NUM_CLASSES = 4
NUM_POINTS = 2**14

DATASET_TRAIN_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TrainFiles/"
DATASET_PREP_TRAIN_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/prepared/TrainFiles/"
DATASET_VALIDATION_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/ValidationFiles/"
DATASET_TEST_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TestFiles/"

CLASS_MAPPING = {
    1: 0,  # original PERSON(1) --> 0
    2: 1,  # original DOG(2) --> 1
    5: 2,  # original BICYCLE(5) --> 2
    6: 3,  # original SPORTSBALL(6) --> 3
}

def load_ascii_cloud_prepared(fname):
    points = []
    labels = []
    instances = []

    with open(fname, 'r') as fd:
        for line in fd.readlines():
            if "//" in line:
                continue

            x, y, t, class_label, instance_label = line.strip().split(' ')
            x, y, t, class_label, instance_label = float(x), float(y), float(t), int(class_label), int(instance_label)

            points.append(np.array([x, y, t], dtype=np.float32))
            labels.append(class_label)
            instances.append(instance_label)

    npPoints = np.array(points, dtype=np.float32)
    npSeg = np.array(labels, dtype=np.uint8)
    npIns = np.array(instances, dtype=np.uint16)

    if len(npIns) != NUM_POINTS:
        raise ValueError("Wrong NUM_POINTS of cloud: ", fname)
    
    return npPoints, npSeg, npIns

class DVSDataset():
    def __init__(self, input_list_txt = 'none', npoints=16384, split='train', batchsize=16):
        random.seed(1337)  # same result every time

        if split not in ['train', 'validation', 'test', 'prepared_train', 'prepared_test']:
            raise ValueError("unknown split")

        self.input_list_txt = input_list_txt
        self.split = split
        self.batch_count = 0
        self.batchsize = batchsize
        
        if npoints != NUM_POINTS:
            raise ValueError("npoints != NUM_POINTS")

        if(input_list_txt == 'none'):
            if(split == 'train'):
                self.files_to_use = glob.glob(os.path.join(DATASET_TRAIN_DIR, "*.csv"))
            elif(split == 'validation'): 
                self.files_to_use = glob.glob(os.path.join(DATASET_VALIDATION_DIR, "*.csv"))
            elif(split == 'test'):
                self.files_to_use = glob.glob(os.path.join(DATASET_TEST_DIR, "*.csv"))
            elif(split == 'prepared_train'):
                self.files_to_use = glob.glob(os.path.join(DATASET_PREP_TRAIN_DIR, "*.csv"))

        else:
            if(split == 'test' or split == 'prepared_test'):
                self.files_to_use = []
                self.files_to_use.append(input_list_txt)
            else:
                self.input_list_txt = input_list_txt
                self.files_to_use = self.get_input_list()

        random.shuffle(self.files_to_use)
        
        
        self.length = len(self.files_to_use)
        self.batch_num = self.length // batchsize

        # --------------------------------------------------------------------------------------------------------------
        # parallel csv read...
        print("Start to read files...")
        if split == 'prepared_train' or split == 'prepared_test':
            self.length = len(self.files_to_use)
            
            if(len(self.files_to_use) == 1):
                points, labels, instances = load_ascii_cloud_prepared(self.files_to_use[0])
            else:
                pool = Pool(processes=None)
                points, labels, instances = zip(*pool.map(load_ascii_cloud_prepared, self.files_to_use))
            
            self.point_list = np.asarray(points)
            self.semantic_label_list = np.asarray(labels)
            self.instance_label_list = np.asarray(instances)
        
        else:
            self.point_list, self.semantic_label_list, self.instance_label_list = prepareData(self.files_to_use)
            self.length = len(self.point_list.shape[0])

        self.batch_num = self.length // batchsize
        
        print(len(self.point_list), len(self.semantic_label_list), len(self.instance_label_list))
        
    def get_input_list(self):
        input_list = [line.strip() for line in open(self.input_list_txt, 'r')]
        input_list = [os.path.join(self.data_root, item) for item in input_list]

        return input_list
    
    # def __len__(self):
    #     return self.length

    # def __getitem__(self, index):
    #     return self.point_list[index], \
    #            self.semantic_label_list[index].astype(np.int32), \
    #            self.labelweights[self.semantic_label_list[0].astype(np.int32)]

    # def get_all(self):
    #     return self.point_list, self.semantic_label_list, self.instance_label_list

    def get_batch(self, data_aug=False):

        points = self.point_list[(self.batch_count*self.batchsize):((self.batch_count+1)*self.batchsize)][:][:]
        sem = self.semantic_label_list[(self.batch_count*self.batchsize):((self.batch_count+1)*self.batchsize)][:][:]
        inst = self.instance_label_list[(self.batch_count*self.batchsize):((self.batch_count+1)*self.batchsize)][:][:]

        self.batch_count = self.batch_count + 1
        if(self.batch_count == self.batch_num):
            self.batch_count = 0
            
        return points, sem, inst

    # def get_length(self):
    #     return self.__len__()
    

# ------------------------------------------------------------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------
    dvsDataset = DVSDataset('/home/klein/neural_networks/jsnet/JSNet_LK/data/train_csv_dvs.txt', split='train', batchsize=16)
    points, sem, inst = dvsDataset.get_batch()
    print(points.shape)
    print(sem.shape)
    print(inst.shape)
    points, sem, inst = dvsDataset.get_batch()
    print(points.shape)
    print(sem.shape)
    print(inst.shape)
    points, sem, inst = dvsDataset.get_batch()
    print(points.shape)
    print(sem.shape)
    print(inst.shape)