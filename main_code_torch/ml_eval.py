import os

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
import torch
import logging
import cv2
from main_code_torch.data_lib import data_load_ml_fold

from main_code_torch.utils import util
from main_code_torch.losses import lovasz_losses
from torch.nn import functional as F
from main_code_torch.data_lib.tgs_agument import *
import matplotlib.pyplot as plt
from main_code_torch.model_include import *
import pandas as pd


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def compute_center_pad(H, W, factor=32):
    if H % factor == 0:
        dy0, dy1 = 0, 0
    else:
        dy = factor - H % factor
        dy0 = dy // 2
        dy1 = dy - dy0

    if W % factor == 0:
        dx0, dx1 = 0, 0
    else:
        dx = factor - W % factor
        dx0 = dx // 2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def test_unaugment_null(prob, shape):
    IMAGE_HEIGHT = shape
    IMAGE_WIDTH = shape
    dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    prob = prob[:, dy0:dy0 + IMAGE_HEIGHT, dx0:dx0 + IMAGE_WIDTH]
    return prob

def make_loader(img_id_path, batch_size, img_size_bp, shuffle=False, transform=None, if_TTA=False):
    return DataLoader(
        dataset=data_load_ml_fold.TGSDS(img_list=img_id_path,transform=transform, if_TTA=if_TTA, img_size_bp=img_size_bp,
                                   if_name=True,resize_mode='pad'),
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

def m_eval(AVE,K,NO_ADD):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    TTA = True
    NO_ADD = NO_ADD   # if eavl the  +40 weights

    img_size_bp = 101
    batch_size = 32
    AVE = AVE
    K=K

    model_name = 'link_dense'
    ex_name = 'link_dense_fold'

    ave_subname = ''.join([str(x) for x in AVE])
    record_name = '{}_k{}_a{}'.format(ex_name,K,ave_subname)


    if TTA:
        record_name = record_name + '_TTA'
    if NO_ADD:
        record_name = record_name+'_NO_ADD'
    print(record_name)




    valid_loader = make_loader(img_id_path='../data/fold_split2/f{}/val.csv'.format(K),
                               batch_size=batch_size, transform=None, if_TTA=TTA, img_size_bp=img_size_bp)
    weights_list = []
    fold_path = 'fold_experiments/{}/k{}/'.format(ex_name, K)
    if NO_ADD:
        for a in AVE:
            w_names = os.listdir(fold_path + 'a{}/weights/'.format(a))
            epochs = [os.path.basename(snapshot).split('_')[1] for snapshot in w_names]
            epochs = np.array(epochs, dtype=np.int32)
            best_epoch = np.max(epochs)
            weight_patht = fold_path + 'a{}/weights/PSPNet_{}'.format(a, best_epoch)

            weights_list.append(weight_patht)
    else:
        for a in AVE:
            w_name = os.listdir(fold_path + 'a{}/best_weight_add/'.format(a))[0]
            weight_patht = fold_path + 'a{}/best_weight_add/'.format(a) + w_name
            weights_list.append(weight_patht)


    preds_valid_all = []


    for weights_path in weights_list:
        with torch.no_grad():
            net, starting_epoch = build_network(weights_path, model_name)
            net.eval()
            val_iterator = tqdm(valid_loader)
            y_valid_ori = []
            preds_valid = []
            all_name = []
            if TTA:
                for vx, vx_f, vy, img_name in val_iterator:
                    vxv, vxv_f, vyv = Variable(vx).cuda(), Variable(vx_f).cuda(), Variable(vy).cuda()
                    out1,_,_ = net(vxv)
                    out1 = F.sigmoid(out1)
                    out1 = torch.squeeze(out1.data.cpu(), dim=1)
                    out1 = out1.numpy()

                    out2,_,_ = net(vxv_f)
                    out2 = F.sigmoid(out2)
                    out2 = torch.squeeze(out2.data.cpu(), dim=1)
                    out2 = out2.numpy()
                    out2_f = np.zeros_like(out2)
                    for t in range(out2.shape[0]):
                        out2_f[t, :, :] = cv2.flip(out2[t, :, :], 1)
                    # out2 = cv2.flip(out2,1)

                    outwith = np.stack([out1, out2_f], axis=-1)
                    out = np.mean(outwith, axis=-1)

                    ## for debug
                    # out = np.where(out>0.5,1,0)
                    #
                    # vy = vy.numpy()
                    #
                    # for x in range(32):
                    #     plt.subplot(211)
                    #     plt.imshow(vy[x,0,:,:])
                    #     plt.subplot(212)
                    #     plt.imshow(out[x,:,:])
                    #     plt.show()

                    vy = np.squeeze(vy.numpy())
                    y_valid_ori.append(vy)
                    preds_valid.append(out)
                    all_name = all_name + img_name
            else:

                for vx, vy in val_iterator:
                    vxv, vyv = Variable(vx).cuda(), Variable(vy).cuda()
                    out = net(vxv)
                    out = F.sigmoid(out)
                    out = torch.squeeze(out.data.cpu(), dim=1)
                    out = out.numpy()

                    ## for debug
                    # out = np.where(out>0.5,1,0)
                    #
                    # vy = vy.numpy()
                    #
                    # for x in range(32):
                    #     plt.subplot(211)
                    #     plt.imshow(vy[x,0,:,:])
                    #     plt.subplot(212)
                    #     plt.imshow(out[x,:,:])
                    #     plt.show()
                    #
                    #
                    # print(out)

                    vy = np.squeeze(vy.numpy())
                    y_valid_ori.append(vy)
                    preds_valid.append(out)

                    # print(vy.shape)

            y_valid_ori = np.concatenate(y_valid_ori, axis=0)
            preds_valid = np.concatenate(preds_valid, axis=0)

            y_valid_ori = test_unaugment_null(y_valid_ori, shape=img_size_bp)  # samples,202,202
            preds_valid = test_unaugment_null(preds_valid, shape=img_size_bp)  # samples,202,202
            #np.save('test',preds_valid)
            preds_valid_all.append(preds_valid)

    preds_valid = np.array(preds_valid_all)
    preds_valid = np.transpose(preds_valid,[1,2,3,0])
    preds_valid = np.mean(preds_valid,axis=-1)


    if y_valid_ori.shape[-1] != 101:
        y_valid_ori2 = np.zeros([y_valid_ori.shape[0], 101, 101])
        preds_valid2 = np.zeros([y_valid_ori.shape[0], 101, 101])
        for s in range(y_valid_ori.shape[0]):
            yt = cv2.resize(y_valid_ori[s, :, :], dsize=(101, 101))
            yt = (yt > 0.5).astype(np.float32)
            y_valid_ori2[s, :, :] = yt

            yt = cv2.resize(preds_valid[s, :, :], dsize=(101, 101))
            # yt = (yt > 0.5).astype(np.float32)
            preds_valid2[s, :, :] = yt

    else:
        y_valid_ori2 = y_valid_ori
        preds_valid2 = preds_valid

    threshold_best = 0.48

    #print(preds_valid2.shape[0])

    pred = np.int32(preds_valid2 > threshold_best)

    iout = util.do_kaggle_metric(predict=pred,truth=y_valid_ori2)


    print('iou best', iout)
    print('threshold_best', threshold_best)
    iof = open('logs/nf_iou_logs.txt', 'a')
    iof.write(record_name + '\t' + str(iout) + '\t' + str(threshold_best) + '\n')
    iof.close()


if __name__ == '__main__':
    import time

    for K in range(1):
        m_eval(K=K,AVE=[0],NO_ADD=False)











