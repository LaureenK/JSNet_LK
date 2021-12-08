# To estimate the mean instance size of each class in training set
import os
import sys
import glob
import numpy as np
from scipy import stats
import argparse

DATASET_TRAIN_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TrainFiles/"
DATASET_VALIDATION_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/ValidationFiles/"
DATASET_TEST_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TestFiles/"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data', help='data dir')
parser.add_argument('--dataset', type=str, default='DVS', help='dataset [S3DIS]')
parser.add_argument('--input_list', type=str, default='data/train_hdf5_file_list_woArea5.txt', help='estimate the mean instance size')
parser.add_argument('--num_cls', type=int, default=4, help='estimate the mean instance size')
parser.add_argument('--out_dir', type=str, default='log5', help='log dir to save mean instance size [model path]')
FLAGS = parser.parse_args()


def estimate(flags):
    num_classes = flags.num_cls
    if flags.dataset == 'DVS':
        train_file_list = glob.glob(os.path.join(DATASET_TRAIN_DIR, "*.csv"))
    else:
        print("Error: Not support the dataset: ", flags.dataset)
        return

    mean_ins_size = np.zeros(num_classes)
    ptsnum_in_gt = [[] for itmp in range(num_classes)]

    for dvs_filename in train_file_list:
        print(dvs_filename)
        cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(dvs_filename)
        cur_data = np.reshape(cur_data, [-1, cur_data.shape[-1]])
        cur_group = np.reshape(cur_group, [-1])
        cur_sem = np.reshape(cur_sem, [-1])

        un = np.unique(cur_group)
        for ig, g in enumerate(un):
            tmp = (cur_group == g)
            sem_seg_g = int(stats.mode(cur_sem[tmp])[0])
            ptsnum_in_gt[sem_seg_g].append(np.sum(tmp))

    for idx in range(num_classes):
        mean_ins_size[idx] = np.mean(ptsnum_in_gt[idx]).astype(np.int)

    print(mean_ins_size)
    np.savetxt(os.path.join(flags.out_dir, 'mean_ins_size.txt'), mean_ins_size)


if __name__ == "__main__":
    estimate(FLAGS)