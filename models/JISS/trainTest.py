import argparse
import os
import socket
import sys
import numpy as np

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from dvs_utils.dataset_dvs import DVSDataset
from log_util import get_logger
from model import *
from clustering import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_works', type=int, default=8, help='Loading data thread [default: 8]')
parser.add_argument('--data_root', default='data', help='data dir [default: data]')
parser.add_argument('--data_type', default='csv', help='data type: numpy or csv [default: csv]')                  #changed
parser.add_argument('--log_dir', default='logs', help='Log dir [default: logs]')
parser.add_argument('--num_point', type=int, default=16384, help='Point number [default: 65536]')                   #changed
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to run [default: 50]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8000]')      #changed!
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=12500, help='Decay step for lr decay [default: 12500]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--input_list', type=str, default='data/train_csv_dvs.txt',
                    help='Input data list file')
parser.add_argument('--restore_model', type=str, default='log/', help='Pretrained model')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size                   #16
NUM_WORKS = FLAGS.num_works                     #8
NUM_POINT = FLAGS.num_point                     #16384
DATA_TYPE = FLAGS.data_type                     #csv
START_EPOCH = FLAGS.start_epoch                 #0
MAX_EPOCH = FLAGS.max_epoch                     #in file --> 1
BASE_LEARNING_RATE = FLAGS.learning_rate        #0.001
GPU_INDEX = FLAGS.gpu                           #0
MOMENTUM = FLAGS.momentum                       #0.9
OPTIMIZER = FLAGS.optimizer                     #adam
DECAY_STEP = FLAGS.decay_step                   #12500
DECAY_STEP = int(DECAY_STEP / (BATCH_SIZE / 24))
DECAY_RATE = FLAGS.decay_rate                   #0.5

DATA_ROOT = FLAGS.data_root                     # ./ -> in file
TRAINING_FILE_LIST = FLAGS.input_list           #data/train_csv_dvs.txt
PRETRAINED_MODEL_PATH = FLAGS.restore_model

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

MAX_NUM_POINT = 16384   #changed
NUM_CLASSES = 4        #changed

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# backup model
os.system('cp model.py {}'.format(LOG_DIR))
os.system('cp train.py {}'.format(LOG_DIR))

logger = get_logger(__file__, LOG_DIR, 'log_train.txt')
logger.info(str(FLAGS) + '\n')


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch,               # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    # Load data
    dataset = DVSDataset(DATA_ROOT, input_list_txt = 'none', split='train',batchsize = BATCH_SIZE)
    #dataset = DVSDataset(DATA_ROOT, TRAINING_FILE_LIST, split='train',batchsize = BATCH_SIZE)
    
    
    # build network and create session
    with tf.Graph().as_default(), tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)

        # Get model and loss
        pred_sem, pred_ins = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
        pred_sem_softmax = tf.nn.softmax(pred_sem)
        pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)

        loss, sem_loss, disc_loss, l_var, l_dist = get_loss(pred_ins, labels_pl, pred_sem_label, pred_sem, sem_labels_pl)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('sem_loss', sem_loss)
        tf.summary.scalar('disc_loss', disc_loss)
        tf.summary.scalar('l_var', l_var)
        tf.summary.scalar('l_dist', l_dist)

        # Get training operator
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        train_op = optimizer.minimize(loss, var_list=tf.trainable_variables(), global_step=batch)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=15)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
            saver.restore(sess, LOAD_MODEL_FILE)
            logger.info("Model loaded in file: %s" % LOAD_MODEL_FILE)

        adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
        sess.run(adam_initializers)

        # ops = {'pointclouds_pl': pointclouds_pl,
        #        'labels_pl': labels_pl,
        #        'sem_labels_pl': sem_labels_pl,
        #        'is_training_pl': is_training_pl,
        #        'loss': loss,
        #        'sem_loss': sem_loss,
        #        'disc_loss': disc_loss,
        #        'l_var': l_var,
        #        'l_dist': l_dist,
        #        'train_op': train_op,
        #        'merged': merged,
        #        'step': batch,
        #        'learning_rate': learning_rate}

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sem_labels_pl': sem_labels_pl,
               'is_training_pl': is_training_pl,
               'pred_ins': pred_ins,                                       #neu
               'pred_sem_label': pred_sem_label,                          #neu
               'loss': loss,
               'sem_loss': sem_loss,
               'disc_loss': disc_loss,
               'l_var': l_var,
               'l_dist': l_dist,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'learning_rate': learning_rate}

        for epoch in range(START_EPOCH, MAX_EPOCH):
            train_one_epoch(sess, ops, train_writer, dataset, epoch)

            # Save the variables to disk.
            if epoch % 5 == 0 or epoch == (MAX_EPOCH - 1):
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch) + '.ckpt'))
                logger.info("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer, dataset, epoch):
    """ ops: dict mapping from string to tf ops """
    print("#### Start Epoch ", epoch, " ####")
    logger.info('Start time')
    is_training = True
    file_size = dataset.get_length()
    num_batches = file_size // BATCH_SIZE

    loss_sum = 0
    acc_sum = 0.0
    diff_sum = 0.0

    max_epoch_len = len(str(MAX_EPOCH))
    num_batches_len = len(str(num_batches))

    for batch_idx in range(num_batches):
        current_data, current_sem, current_label = dataset.get_batch(False)

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['sem_labels_pl']: current_sem,
                     ops['is_training_pl']: is_training}

        pred_ins_val, pred_sem_label_val, summary, step, lr_rate, _, loss_val, sem_loss_val, disc_loss_val, l_var_val, l_dist_val = sess.run(
            [ops['pred_ins'], ops['pred_sem_label'], ops['merged'], ops['step'], ops['learning_rate'], ops['train_op'], ops['loss'], ops['sem_loss'],
             ops['disc_loss'], ops['l_var'], ops['l_dist']], feed_dict=feed_dict)
        
        #print("Pred_ins_val: ", pred_ins_val[0,0,:], " pred_sem_label_val: ", pred_sem_label_val[0,0])
        
        i = 0
        sum_acc = 0
        sum_diff = 0
        while i < BATCH_SIZE:
            sem1 = pred_sem_label_val[i]
            sem2 = current_sem[i]
            ins1 = current_label[i]
            ins1num = len(np.unique(ins1)) #falsch nicht unique!!!!!!!!!!!!!
            ins2 = pred_ins_val[i]
            #print("block1: ", block1.shape, " Block2: ", block2.shape)

            right_pred = np.count_nonzero(sem1==sem2)
            sum_acc = float((right_pred/(NUM_POINT)))

            #print("Right: ", right_pred, " from: 16384 accuracy: ", float((right_pred/(NUM_POINT) * 100)))

            bandwidth = 4
            num_clusters, labels, cluster_centers = cluster(ins2, bandwidth)
            sum_diff += abs(ins1num - num_clusters)
            print("Right num of instances: ", ins1num, " Predicted num: ", num_clusters)

            i = i+1

        acc_sum += float((sum_acc/(BATCH_SIZE)))
        diff_sum += float((sum_diff/(BATCH_SIZE)))
        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if batch_idx % 50 == 0 and batch_idx:
            logger_info = "epoch: {1:0{0}d}/{2}; batch_num: {4:0{3}d}/{5}; lr_rate: {6:.6f}; loss: {7:.2f}; " \
                          "sem_loss: {8:.2f}; disc_loss: {9:.2f}; l_var: {10:.2f}; l_dist: {11:.2f};"

            logger.info(logger_info.format(max_epoch_len, epoch, MAX_EPOCH, num_batches_len, batch_idx, num_batches,
                                           lr_rate, loss_val, sem_loss_val, disc_loss_val, l_var_val, l_dist_val))

    if(loss_sum == 0):
        logger.info('mean loss: %f' % (loss_sum))
    else:
        logger.info('mean loss: %.2f' % (loss_sum / float(num_batches)))
        logger.info('Semantic mean accuracy: %.2f' % ((acc_sum / float(num_batches))*100))
        logger.info('Instance mean difference: %.2f' % (diff_sum / float(num_batches)))


if __name__ == "__main__":
    train()
