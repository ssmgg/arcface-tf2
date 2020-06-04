from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import random
import glob
from mtcnn import MTCNN
import math
from SCLab import distance

from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
import time

flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

    ckpt_path = tf.train.latest_checkpoint('../checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    if FLAGS.img_path:
        str1 = 'this'
        str2 = 'this1'
        #print(str1 in str2)

        file_dir = "newTWICE_id"
        output = glob.glob(file_dir + "/*.npy")

        for i in range(100):
            sampleList = random.sample(output, 2)

            embedding1 = np.load(sampleList[0])
            embedding2 = np.load(sampleList[1])
            dist = distance(embedding1,embedding2,1)

            str1 = sampleList[0].split("\\")[1].split(".")[0]
            str2 = sampleList[1].split("\\")[1].split(".")[0]
            print(str1)
            if dist<0.4:
                print(str1 + " and " + str2 + "is same")
                print(dist)

                #             tp = tp + 1
            else:
                print(str1 + " and " + str2 + "is diff")
                print(dist)
                    #             tf = tf + 1


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
