# -*- coding: utf-8 -*-
import os
import glob
import random
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--inputpath', default='/bigdata_hdd/klein/FrKlein_PoC/data/TrainFiles/')
parser.add_argument('--outputpath', default='/bigdata_hdd/klein/FrKlein_PoC/data/prepared/TrainFiles/')
#parser.add_argument('--inputpath', default='E:/Users/Laureen/Documents/Test1/')
#parser.add_argument('--outputpath', default='E:/Users/Laureen/Documents/Test2/')
FLAGS = parser.parse_args()  

INPUT = FLAGS.inputpath                   
OUTPUT = FLAGS.outputpath                   
  
NUM_CLASSES = 4
NUM_POINTS = 2**14

#DATASET_TRAIN_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TrainFiles/"
#DATASET_VALIDATION_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/ValidationFiles/"
#DATASET_TEST_DIR = "/bigdata_hdd/klein/FrKlein_PoC/data/TestFiles/"

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
    npIns = np.array(instances, dtype=np.uint32)

    npPoints, npSeg, npIns = unison_shuffled_copies(npPoints, npSeg, npIns)
 
    return npPoints, npSeg, npIns

def upscale(points, labels, instances):
    # if len(points) > NUM_POINTS:
    #     raise RuntimeError("no matching config...!")

    if len(points) < NUM_POINTS:
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

def create_two_x(points, labels, instances, start= 0, end= 640):
    small_points1 = []
    small_labels1 = []
    small_instances1 = []

    small_points2 = []
    small_labels2 = []
    small_instances2 = []

    i = 0
    while i < len(points):
        point = points[i]
        if(point[0] < (start + ((end-start) / 2))):
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
    left = False
    right = False

    if small_points1:
        small_points1, small_labels1, small_instances1 = upscale(small_points1, small_labels1, small_instances1)
        small_points3.append(small_points1)
        small_labels3.append(small_labels1)
        small_instances3.append(small_instances1)
        left = True
    
    if small_points2:
        small_points2, small_labels2, small_instances2 = upscale(small_points2, small_labels2, small_instances2)
        small_points3.append(small_points2)
        small_labels3.append(small_labels2)
        small_instances3.append(small_instances2)
        right = True

    return small_points3, small_labels3, small_instances3, left, right

def create_two_y(points, labels, instances, start = 0, end = 768):
    small_points1 = []
    small_labels1 = []
    small_instances1 = []

    small_points2 = []
    small_labels2 = []
    small_instances2 = []

    i = 0
    while i < len(points):
        point = points[i]
        if(point[1] < (start + ((end-start) / 2))):
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

    top = False
    down = False

    if small_points1:
        small_points1, small_labels1, small_instances1 = upscale(small_points1, small_labels1, small_instances1)
        small_points3.append(small_points1)
        small_labels3.append(small_labels1)
        small_instances3.append(small_instances1)
        top = True
    
    if small_points2:
        small_points2, small_labels2, small_instances2 = upscale(small_points2, small_labels2, small_instances2)
        small_points3.append(small_points2)
        small_labels3.append(small_labels2)
        small_instances3.append(small_instances2)
        down = True

    return small_points3, small_labels3, small_instances3, top, down

def downscale(points, labels, instances, x=True, depth = 1, left=True, top=True):
    small_points = []
    small_labels = []
    small_instances = []

    if x == True:
        if depth == 1 and left == True:
            small_points1, small_labels1, small_instances1, left1, right1 = create_two_x(points, labels, instances)
        elif depth == 2 and left == True:
            small_points1, small_labels1, small_instances1, left1, right1 = create_two_x(points, labels, instances, 0,320)
        elif depth == 2 and left == False:
            small_points1, small_labels1, small_instances1, left1, right1 = create_two_x(points, labels, instances, 320,640)

        # print("After downscale X: ", len(small_points1[0]))
        # print("After downscale X: ", len(small_points1[1]))
        
        if left1:
            if len(small_points1[0]) > NUM_POINTS:
                small_points2, small_labels2, small_instances2 = downscale(small_points1[0], small_labels1[0], small_instances1[0], False,depth,True,top)
                i = 0
                while i < len(small_points2):
                    small_points.append(small_points2[i])
                    small_labels.append(small_labels2[i])
                    small_instances.append(small_instances2[i])
                    i = i +1
            else:
                small_points.append(small_points1[0])
                small_labels.append(small_labels1[0])
                small_instances.append(small_instances1[0])

        if right1:
            if len(small_points1[1]) > NUM_POINTS:
                small_points2, small_labels2, small_instances2 = downscale(small_points1[1], small_labels1[1], small_instances1[1], False,depth,False,top)
                i = 0
                while i < len(small_points2):
                    small_points.append(small_points2[i])
                    small_labels.append(small_labels2[i])
                    small_instances.append(small_instances2[i])
                    i = i +1
            else:
                small_points.append(small_points1[1])
                small_labels.append(small_labels1[1])
                small_instances.append(small_instances1[1])
    else:
        if depth == 1 and top == True:
            small_points1, small_labels1, small_instances1, top1, down1 = create_two_y(points, labels, instances)
        elif depth == 2 and top == True:
            small_points1, small_labels1, small_instances1, top1, down1 = create_two_y(points, labels, instances, 0, 384)
        elif depth == 2 and top == False:
            small_points1, small_labels1, small_instances1, top1, down1 = create_two_y(points, labels, instances, 384, 768)
        

        small_points1, small_labels1, small_instances1, top1, down1 = create_two_y(points, labels, instances)
        # print("After downscale Y: ", len(small_points1[0]))
        # print("After downscale Y: ", len(small_points1[1]))

        if top1:
            if len(small_points1[0]) > NUM_POINTS:
                small_points2, small_labels2, small_instances2 = downscale(small_points1[0], small_labels1[0], small_instances1[0], True, depth+1,left,True)
                i = 0
                while i < len(small_points2):
                    small_points.append(small_points2[i])
                    small_labels.append(small_labels2[i])
                    small_instances.append(small_instances2[i])
                    i = i +1
            else:     
                small_points.append(small_points1[0])
                small_labels.append(small_labels1[0])
                small_instances.append(small_instances1[0])
        
        if down1:
            if len(small_points1[1]) > NUM_POINTS:
                small_points2, small_labels2, small_instances2 = downscale(small_points1[1], small_labels1[1], small_instances1[1], True, depth+1,left,False)
                i = 0
                while i < len(small_points2):
                    small_points.append(small_points2[i])
                    small_labels.append(small_labels2[i])
                    small_instances.append(small_instances2[i])
                    i = i +1
            else:     
                small_points.append(small_points1[1])
                small_labels.append(small_labels1[1])
                small_instances.append(small_instances1[1])


    return small_points, small_labels, small_instances

def mapInstance(points, labels, instances):
    instances = instances + 100

    i = 0
    while i < len(labels):
        num = instances[i]
        digits = len(str(num))
        addition = 10**digits

        if (labels[i] == 0):
            instances[i] = instances[i] + (4 * addition)
        elif (labels[i] == 1):
            instances[i] = instances[i] + (1 * addition)
        elif (labels[i] == 2):
            instances[i] = instances[i] + (2 * addition)
        elif (labels[i] == 3):
            instances[i] = instances[i] + (3 * addition)

        i = i + 1

    un = np.unique(instances)
    count = len(un)

    i = 0
    while i < count:
        instances[instances == un[i]] = i
        i = i + 1

    un = np.unique(instances)

    return points, labels, instances

if __name__ == "__main__":
    INPUTLIST = glob.glob(os.path.join(INPUT, "*.csv"))
    print(len(INPUTLIST))
    output_num = 0

    i=0
    while i < len(INPUTLIST):
        points, labels, instances = load_and_upscale(INPUTLIST[i])
        points, labels, instances = mapInstance(points, labels, instances)

        points = np.asarray(points)
        labels = np.asarray(labels)
        instances = np.asarray(instances)


        if(len(points) > NUM_POINTS):
            small_points, small_labels, small_instances = downscale(points, labels, instances)
            small_points = np.asarray(small_points)
            small_labels = np.asarray(small_labels)
            small_instances = np.asarray(small_instances)

            j = 0

            while j < small_points.shape[0]:
                points1 = small_points[j]
                labels1 = small_labels[j]
                instances1 = small_instances[j]

                labels1 = np.reshape(labels1, (len(labels1),1))
                instances1 = np.reshape(instances1,(len(instances1),1))
                all = np.append(points1, labels1, axis=1)
                all = np.append(all, instances1, axis=1)

                name = OUTPUT + str(output_num) + ".csv"
                print(name)
                np.savetxt(name, all, delimiter=" ", fmt='%d %d %.10f %d %d')

                output_num = output_num + 1

                j = j + 1


        else:
            labels = np.reshape(labels, (len(labels),1))
            instances = np.reshape(instances,(len(instances),1))
            all = np.append(points, labels, axis=1)
            all = np.append(all, instances, axis=1)

            name = OUTPUT + str(output_num) + ".csv"
            print(name)
            np.savetxt(name, all, delimiter=" ",fmt='%d %d %.10f %d %d')
            
            output_num = output_num + 1
            

        i = i + 1

    print("finish")