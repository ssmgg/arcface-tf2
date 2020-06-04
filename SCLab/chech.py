from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf

import glob
from mtcnn import MTCNN

from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
import time
from SCLab import distance

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

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    if FLAGS.img_path:

        print("Check Start!!!")

        file_dir = "C:/Users/smgg/Desktop/dataset/superjunior/all3.jpeg"
        npy_dir = "/SCLab/newTWICE_id/*,npy"

        img_list = glob.glob(file_dir)
        npy_list = glob.glob(npy_dir)

        for img_name in img_list:
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
            detector = MTCNN()
            data_list = detector.detect_faces(img)

            for data in data_list:
                xmin, ymin, width, height = data['box']
                xmax = xmin + width
                ymax = ymin + height

                face_image = img[ymin:ymax, xmin: xmax, :]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

                # cv2.imshow('', face_image)
                # cv2.waitKey(0)

                img_resize = cv2.resize(face_image, (cfg['input_size'], cfg['input_size']))
                img_resize = img_resize.astype(np.float32) / 255.
                if len(img_resize.shape) == 3:
                    img_resize = np.expand_dims(img_resize, 0)
                embeds = l2_norm(model(img_resize))

                i = 0
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                for npy_name in npy_list:

                    name_embeds = np.load(npy_name)
                    # print(1)

                    if distance(embeds, name_embeds,1) < 0.37:
                        i = i + 1
                        name = npy_name.split('/')[5].split('\\')[1].split('.npy')[0]

                        cv2.putText(img, name, (xmin, ymin - 15*(i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,cv2.LINE_AA)


                    # else:
                    #     cv2.putText(img, "Unknown", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
