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

        print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))

        # file_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/data/AFDB_masked_face_dataset/*"
        file_dir = "C:/Users/smgg/Desktop/dataset/superjunior/*.jpg"
        # detector = MTCNN()
        i = 0
        img_list = glob.glob(file_dir)
        for i_list in img_list:
            img = cv2.cvtColor(cv2.imread(i_list), cv2.COLOR_BGR2RGB)
            detector = MTCNN()
            data_list = detector.detect_faces(img)

            for data in data_list:
                xmin, ymin, width, height = data['box']
                xmax = xmin + width
                ymax = ymin + height

                face_image = img[ymin:ymax, xmin: xmax, :]
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('', face_image)

                print(i_list)

                img_resize = cv2.resize(face_image, (cfg['input_size'], cfg['input_size']))
                cv2.imwrite("./{}.jpg".format(i), img_resize)
                cv2.imshow('',img_resize)
                cv2.waitKey(0)
                i = i + 1

                cv2.putText(img, "0", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

                img_resize = img_resize.astype(np.float32) / 255.
                if len(img_resize.shape) == 3:
                    img_resize = np.expand_dims(img_resize, 0)
                embeds = l2_norm(model(img_resize))

                #print('./newTWICE_id/{}.npy'.format(i_list.split('/')[6].split('\\')[1].split('.jpg')[0]))
                np.save('./newTWICE_id/{}.npy'.format(i_list.split('/')[5].split('\\')[1].split('.jpg')[0]), embeds)
    else:
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])

        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

        print("[*] Perform Evaluation on AgeDB30...")
        acc_agedb30, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
            agedb_30_issame, is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

        print("[*] Perform Evaluation on CFP-FP...")
        acc_cfp_fp, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass