import argparse
import os
import socket
import sys
import glob
import time

import numpy as np
import tensorflow as tf
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from model import *
from test_utils import *
from log_util import get_logger
from clustering import cluster
from dvs_utils.dataset_dvs import DVSDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/klein/neural_networks/jsnet/JSNet_LK/logs/train_dvs_6/epoch_99.ckpt', help='Path of model')
parser.add_argument('--input_path', type=str, default="/bigdata_hdd/klein/FrKlein_PoC/data/prepared/TestFiles/", help='Path of test files')
parser.add_argument('--output_path', type=str, default='/home/klein/neural_networks/jsnet/JSNet_LK/logs/test_dvs_time/result/', help='Result path')
parser.add_argument('--log_path', type=str, default='/home/klein/neural_networks/jsnet/JSNet_LK/logs/test_dvs_Time/', help='Log path')
FLAGS = parser.parse_args() 

GPU_INDEX = 0
NUM_POINT = 2**14
BATCH_SIZE = 1
NUM_CLASSES = 4
BANDWIDTH = 0.6
MODEL_PATH = FLAGS.model_path

#don't forget to change the radius in model.py depending to the model

DATASET_TEST_DIR = FLAGS.input_path
ROOM_PATH_LIST = glob.glob(os.path.join(DATASET_TEST_DIR, "*.csv"))
len_pts_files = len(ROOM_PATH_LIST)

LOG_DIR = FLAGS.log_path
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

OUTPUT_PATH = FLAGS.output_path
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

logger = get_logger(__file__, LOG_DIR, 'log_inference.txt')
logger.info(str(FLAGS) + '\n')

def safeFile(pts, gt_sem, gt_group, pred_sem, labels, softmax, file_path):
    filename = file_path.split('/')[-1].split('.')[0]

    with open(file_path, 'r') as fd:
        head = fd.readlines()[0]
    
    gt_sem = np.reshape(gt_sem, (len(pred_sem),1))
    gt_group = np.reshape(gt_group,(len(labels),1))
    sem_labels = np.reshape(pred_sem, (len(pred_sem),1))
    instances = np.reshape(labels,(len(labels),1))
    softmax = np.reshape(softmax,(len(softmax),1))

    all = np.append(pts, gt_sem, axis=1)
    all = np.append(all, gt_group, axis=1)
    all = np.append(all, sem_labels, axis=1)
    all = np.append(all, instances, axis=1)
    all = np.append(all, softmax, axis=1)

    name = OUTPUT_PATH + filename + ".csv"
    #print("Save ", name)
    
    np.savetxt(name, all, delimiter=" ", header=head, fmt='%d %d %.10f %d %d %d %d %.3f', comments='//')

def getSoftmaxForSem(pred_sem,sem_softmax):
    softmax = []
    for i in range(len(pred_sem)):
        softmax.append(sem_softmax[i, pred_sem[i]])
    
    softmax = np.array(softmax)

    return softmax

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model
            pred_sem, pred_ins = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
            pred_sem_softmax = tf.nn.softmax(pred_sem)
            pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)

            loader = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        is_training = False

        # Restore variables from disk.
        loader.restore(sess, MODEL_PATH)
        logger.info("Model restored from {}".format(MODEL_PATH))

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sem_labels_pl': sem_labels_pl,
               'is_training_pl': is_training_pl,
               'pred_ins': pred_ins,
               'pred_sem_label': pred_sem_label,
               'pred_sem_softmax': pred_sem_softmax}

        for file_idx in range(len_pts_files):
        #for file_idx in range(1):
            file_path = ROOM_PATH_LIST[file_idx]
            #file_path = "/bigdata_hdd/klein/FrKlein_PoC/data/prepared/TestFiles/102.csv"

            dataset = DVSDataset(input_list_txt = file_path, split='prepared_test')
            cur_data, cur_sem, cur_group = dataset.get_all()

            logger.info("Processsing: File [%d] of [%d]" % (file_idx, len_pts_files))

            pts = cur_data
            gt_group = cur_group
            gt_sem = cur_sem

            feed_dict = {ops['pointclouds_pl']: np.expand_dims(pts, 0),
                        ops['labels_pl']: np.expand_dims(gt_group, 0),
                        ops['sem_labels_pl']: np.expand_dims(gt_sem, 0),
                        ops['is_training_pl']: is_training}

            start_time = time.time()
            pred_ins_val, pred_sem_label_val, pred_sem_softmax_val = sess.run(
                [ops['pred_ins'], ops['pred_sem_label'], ops['pred_sem_softmax']], feed_dict=feed_dict)
            
            duration = time.time() - start_time
            print("Time: ")
            print(duration)

            #instance
            pred_val = np.squeeze(pred_ins_val, axis=0)
            #sem label
            pred_sem = np.squeeze(pred_sem_label_val, axis=0)
            #softmax
            #np.set_printoptions(suppress=True)
            pred_sem_softmax = np.squeeze(pred_sem_softmax_val, axis=0)
            sem_softmax = pred_sem_softmax.reshape([-1, NUM_CLASSES])
            softmax = getSoftmaxForSem(pred_sem,sem_softmax)

            bandwidth = BANDWIDTH
            num_clusters, labels, cluster_centers = cluster(pred_val, bandwidth)

            #safeFile(pts, gt_sem, gt_group, pred_sem, labels, softmax, file_path)


if __name__ == "__main__":
    test()
