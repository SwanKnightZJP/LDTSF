import os
import argparse
import torch
from networks.vnet_ldc import VNet
from test_util import test_all_case, save_all_cases

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='LDC_Best', help='model_name')   # TODO  the pretrained LDC model
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')


FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}".format(FLAGS.model)   # model/DTC_16labels_pretrain

num_classes = 2

test_save_path = os.path.join(snapshot_path, "pseudo/")
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/pseudo.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
              image_list]


def test_calculate_metric():
    net = VNet(n_channels=1, n_classes=num_classes-1,
               normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(
        snapshot_path, 'best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    save_all_cases(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path,
                               nms=FLAGS.nms)

if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000
    print(metric)


