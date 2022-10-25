import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_ldc import VNet

from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_three_sdf
from test_util import  test_show_all_case_t


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/LDC_with_consis_weight', help='model_name')
parser.add_argument('--exp_tmp', type=str,
                    default='LA/LDC_Tmp', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum iter number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')

parser.add_argument('--val_num', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--val_path', type=str,
                    default='../data/2018LA_Seg_Training Set', help='Name of Experiment')

parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0,
                    help='apply NMS post-procssing?')
parser.add_argument('--N_pth', type=int, default=3,
                    help='last N_pth')


args = parser.parse_args()

train_data_path = args.root_path
# snapshot_path = "../model/" + args.exp + \
#     "_{}labels_beta_{}/".format(
#         args.labelnum, args.beta)

snapshot_path = "../model/" + args.exp
tmp_model_pth = "../model/" + args.exp_tmp

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))  # 4 * 1 = 4
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs  # 2

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if not os.path.exists(tmp_model_pth):
        os.makedirs(tmp_model_pth)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes-1,           # TODO ori = 1 with sigmoid
                   normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))            # 0-16
    unlabeled_idxs = list(range(labelnum, 80))      # 16-80
    batch_sampler = TwoStreamBatchSampler(          #
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)  # 16 80 4 2

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    test_save_path = os.path.join(snapshot_path, "test/")
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    print(test_save_path)

    dice_tmp=0
    for epoch_num in iterator:
        time1 = time.time()

        # train
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs_tanh, outputs, out_body, out_detail = model(volume_batch)  # 4 1 112 112 80
            outputs_soft = torch.sigmoid(outputs)

            body_detail = out_body + out_detail

            # calculate the loss
            with torch.no_grad():
                gt_dis, gt_body, gt_detail = compute_three_sdf(label_batch[:].cpu().numpy(), outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
                gt_body = torch.from_numpy(gt_body).float().cuda()
                gt_detail = torch.from_numpy(gt_detail).float().cuda()
                # tmp = 'tmp'

            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)                  # supervised loss
            loss_body = mse_loss(out_body[:labeled_bs, 0, ...], gt_body)
            loss_detail = mse_loss(out_detail[:labeled_bs, 0, ...], gt_detail)

            loss_seg = ce_loss(                                                             # supervised loss
                outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(                                               # supervised loss
                outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)

            dis_to_mask = torch.sigmoid(-1500*outputs_tanh)
            body_to_mask = torch.tanh(1500*out_body)
            detail_to_mask = out_detail  # TODO

            body_detail_mask = torch.tanh(body_detail)

            consistency_loss_dis = torch.mean((dis_to_mask - outputs_soft) ** 2)                # unsupervised data used
            consistency_loss_body = torch.mean((body_to_mask - outputs_soft) ** 2)                # unsupervised data used
            consistency_loss_detail = torch.mean((detail_to_mask - outputs_soft) ** 2)                # unsupervised data used

            consistency_loss_body_detail =  torch.mean((body_detail_mask - outputs_soft) ** 2)

            # supervised_loss = loss_seg_dice + args.beta * (loss_sdf + loss_body + loss_detail) / 3
            supervised_loss = loss_seg_dice + args.beta * (2 * loss_sdf + loss_body + loss_detail) / 4

            consistency_weight = get_current_consistency_weight(iter_num//150)

            # loss = supervised_loss + consistency_weight * (consistency_loss_dis  + consistency_loss_body + consistency_loss_detail) / 3
            loss = supervised_loss + consistency_weight * (consistency_loss_dis + consistency_loss_body_detail) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(
                outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss_dis, iter_num)

            writer.add_scalar('aux_loss/consistency_loss_body',
                              consistency_loss_body, iter_num)
            writer.add_scalar('aux_loss/consistency_loss_detail',
                              consistency_loss_detail, iter_num)
            writer.add_scalar('aux_loss/loss_body',
                              loss_body, iter_num)
            writer.add_scalar('aux_loss/loss_detail',
                              loss_detail, iter_num)

            writer.add_scalar('aux_loss/consistency_loss_body_detail',
                              consistency_loss_body_detail, iter_num)


            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
                (iter_num, loss.item(), consistency_loss_dis.item(), loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(   # b=4, c=1 x=112 y=112 z=80
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = body_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/Body2Mask', grid_image, iter_num)

                image = detail_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/Detail2Mask', grid_image, iter_num)


                image = outputs_tanh[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = gt_dis[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap', grid_image, iter_num)

                image = gt_body[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/GT_BodyMap', grid_image, iter_num)

                image = gt_detail[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/GT_DetailMap', grid_image, iter_num)

                image = out_body[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/BodyMap', grid_image, iter_num)

                image = out_detail[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/DetailMap', grid_image, iter_num)

                image = body_detail_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('aux_train/body_detail_map', grid_image, iter_num)
                
            # # change lr (mile stone)
            # if iter_num % 2500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 2500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_

            # change lr (poly policy)
            lr_ = base_lr * (1 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()

        # save last N epoch models
        if epoch_num > max_epoch - args.N_pth - 1:
            save_mode_path = os.path.join(
                    tmp_model_pth, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

        # val every 15 epoches
        if epoch_num % 15 == 0:

            model.eval()
            torch.cuda.empty_cache()

            with open(args.val_path + '/test.list', 'r') as f:
                image_list = f.readlines()
            image_list = [args.val_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                          image_list]

            avg_metric, pred, score, label = test_show_all_case_t(model, image_list, num_classes=num_classes,
                                       patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                       save_result=False, test_save_path=test_save_path,
                                       metric_detail=args.detail, nms=args.nms)

            # # to save the best result
            # if dice_tmp < avg_metric[0]:
            #     dice_tmp = avg_metric[0]
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'best' + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)

            writer.add_scalar('val_results/Dice', avg_metric[0], iter_num)
            writer.add_scalar('val_results/Jaccard', avg_metric[1], iter_num)
            writer.add_scalar('val_results/95HD', avg_metric[2], iter_num)
            writer.add_scalar('val_results/ASD', avg_metric[3], iter_num)

            pred = torch.tensor(pred.astype(np.float32))[:,:,20:61:10].unsqueeze(0).permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(pred, 5, normalize=False)
            writer.add_image('val_results/pred', grid_image, iter_num)

            score_1 = torch.tensor(score)[0, :, :, 20:61:10].unsqueeze(0).permute(
                3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(score_1, 5, normalize=False)
            writer.add_image('val_results/score1', grid_image, iter_num)

            score_2 = torch.tensor(label)[:, :, 20:61:10].unsqueeze(0).permute(
                3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(score_2, 5, normalize=False)
            writer.add_image('val_results/label', grid_image, iter_num)

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
