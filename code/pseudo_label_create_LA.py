import os
import argparse
import shutil

import torch
from networks.vnet_ldc import VNet
from test_util import save_all_cases_t
from tqdm import tqdm

import numpy as np

import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--pseudo_path', type=str,
                    default='../data/2018LA_Seg_PseudoTraining Set', help='Name of Experiment')

parser.add_argument('--model', type=str,
                    default='/LA/LDC_Tmp', help='model_name')   # TODO  the pretrained tmp LDC model
parser.add_argument('--model_final', type=str,
                    default='/LA/LDC_Best', help='model_name')   # TODO  the pretrained best LDC model

parser.add_argument('--tmp_pseudo', type=str,
                    default='LA_pseudo_tmp', help='model_name')   # TODO  the tmp_pseudo_pth model
parser.add_argument('--final_pseudo', type=str,
                    default='LA_pseudo_final', help='model_name')   # TODO  the tmp_pseudo_pth model

parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')
parser.add_argument('--N_pth', type=int, default=3,
                    help='last N_pth')
parser.add_argument('--threshold', type=float, default=0.02,
                    help='pseudo label threshold')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}".format(FLAGS.model_final)   # model/LDC_Best
tmp_snapshot_path = "../model/{}".format(FLAGS.model)     # model/LDC_tmp

tmp_pseudo_pth = "../data/{}".format(FLAGS.tmp_pseudo)        # data/LA_pseudo_tmp
final_pseudo_pth = "../data/{}".format(FLAGS.final_pseudo)    # data/LA_pseudo_final

num_classes = 2

with open(FLAGS.root_path + '/pseudo.list', 'r') as f:
    image_lists = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
              image_lists]


def call_mse(best, target):
    return np.mean((best - target) ** 2)


def save_single_model(model_pth, save_pth):
    net = VNet(n_channels=1, n_classes=num_classes - 1,
               normalization='batchnorm', has_dropout=False).cuda()
    net.load_state_dict(torch.load(model_pth))
    print("init weight from {}".format(model_pth))
    net.eval()

    save_all_cases_t(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=save_pth,
                               nms=FLAGS.nms)


def save_tmp_pseudo():
    if os.path.exists(tmp_pseudo_pth):
        shutil.rmtree(tmp_pseudo_pth)
    if not os.path.exists(tmp_pseudo_pth):
        os.makedirs(tmp_pseudo_pth)

    for i, model in tqdm(enumerate([f for f in os.listdir(tmp_snapshot_path) if f.endswith('pth')])):
        tmp_save_path = os.path.join(tmp_pseudo_pth, str(i))
        if not os.path.exists(tmp_save_path):
            os.makedirs(tmp_save_path)
        print(tmp_save_path)
        model_pth = os.path.join(tmp_snapshot_path, model)
        save_single_model(model_pth, tmp_save_path)


def save_final_pseudo():

    if os.path.exists(final_pseudo_pth):
        shutil.rmtree(final_pseudo_pth)
    if not os.path.exists(final_pseudo_pth):
        os.makedirs(final_pseudo_pth)

    pseudo_list = os.listdir(tmp_pseudo_pth)

    if len(pseudo_list) - 1 == 0:    # if exist only one pseudo_label_folder
        shutil.copytree(os.path.join(tmp_pseudo_pth, pseudo_list[0]), final_pseudo_pth)

    else:
        conf0 = 1
        print("start cal the MSE and save the effective pseudo labels")
        for i in range(len(pseudo_list) - 1):
            c_dir = os.path.join(tmp_pseudo_pth, str(i))
            t_dir = os.path.join(tmp_pseudo_pth, str(i+1))
            tmp = []
            for j, name in tqdm(enumerate(os.listdir(c_dir))):
                pth_c = c_dir + '/' + name + '/mri_norm2.h5'
                pth_t = t_dir + '/' + name + '/mri_norm2.h5'

                h5fc = h5py.File(pth_c, 'r')
                s_label_c = h5fc['score'][0, :]
                h5ft = h5py.File(pth_t, 'r')
                s_label_t = h5ft['score'][0, :]

                if call_mse(s_label_c, s_label_t) < FLAGS.threshold:
                    tmp.append(1)
                else:
                    tmp.append(0)

            conf0 = conf0 * np.asarray(tmp)

        for m, h5name in enumerate(os.listdir(os.path.join(tmp_pseudo_pth, str(0)))):
            if conf0[m] != 0:
                to_save_pth = final_pseudo_pth + '/' + h5name
                if os.path.exists(to_save_pth):
                    shutil.rmtree(to_save_pth)
                shutil.copytree((tmp_pseudo_pth + '/' + pseudo_list[-1] + '/' + h5name), to_save_pth)
            else:
                continue


def made_pseudo_set():
    # made pseudo_train_list 16 from train set and n from made_pseudo
    with open(FLAGS.root_path + '/train.list', 'r') as f:
        all_lists = f.readlines()
    pseudo_lists = all_lists[0:16]

    print('Moving supervised labels to PseudoSet')
    # move sup_label to pseudo_set
    for k, h5name in tqdm(enumerate(pseudo_lists)):
        to_save_pth = FLAGS.pseudo_path + '/' + h5name.replace('\n', '')
        if os.path.exists(to_save_pth):
            shutil.rmtree(to_save_pth)
        shutil.copytree((FLAGS.root_path + '/' + h5name.replace('\n', '')), to_save_pth)

    print('Moving Pseudo labels to PseudoSet')
    # expend the pseudo_list by pseudo_labels and move the pseudo labels
    for _, pseudo_file in tqdm(enumerate(os.listdir(final_pseudo_pth))):
        pseudo_lists.append(pseudo_file+'\n')
        to_save_pth = FLAGS.pseudo_path + '/' + pseudo_file
        if os.path.exists(to_save_pth):
            shutil.rmtree(to_save_pth)
        shutil.copytree((final_pseudo_pth + '/' + pseudo_file), to_save_pth)

    if os.path.exists(FLAGS.pseudo_path + '/train.list'):
        os.remove(FLAGS.pseudo_path + '/train.list')
    with open(FLAGS.pseudo_path + '/train.list', 'w') as pf:
        for i in range(len(pseudo_lists)):
            pf.writelines(pseudo_lists[i])


if __name__ == '__main__':
    save_tmp_pseudo()
    save_final_pseudo()  #
    made_pseudo_set()
    print('over')


