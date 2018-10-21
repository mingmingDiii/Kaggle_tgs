import os

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import click
import numpy as np
import torch
import logging

from main_code_torch.data_lib import data_load_ml_fold

from main_code_torch.utils import util
from main_code_torch.losses import lovasz_losses
from torch.nn import functional as F
from main_code_torch.data_lib.tgs_agument import *
from main_code_torch.utils.crf import dense_crf_tgs
import time


from main_code_torch.model_include import *
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        epoch = os.path.basename(snapshot).split('_')[-1]
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch



# IMAGE_HEIGHT = 101
# IMAGE_WIDTH = 101
def compute_center_pad(H,W, factor=32):

    if H%factor==0:
        dy0,dy1=0,0
    else:
        dy  = factor - H%factor
        dy0 = dy//2
        dy1 = dy - dy0

    if W%factor==0:
        dx0,dx1=0,0
    else:
        dx  = factor - W%factor
        dx0 = dx//2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1
def test_unaugment_null(prob,img_size_bp):
    IMAGE_HEIGHT = img_size_bp
    IMAGE_WIDTH = img_size_bp
    dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    prob = prob[:,dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    return prob


def m_test():
    img_size_bp = 101
    batch_size = 32
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    TTA = True


    KFOLDS = [2,0,1,3,4,5,6,7,8,9]
    AVE = [0,1,2,3,4]

    ex_name = 'link_dense_fold'
    mode_name = 'link_dense'

    out_file_name = 'link_dense_fold'
    mresult_save = 'link_dense_fold'
    mresult_save_path = 'm_results/{}/'.format(mresult_save)
    if not os.path.exists(mresult_save_path):
        os.mkdir(mresult_save_path)

    if TTA:
        out_file_name = out_file_name+'_TTA'

    threshold_best = 0.48



    def make_loader(batch_size,img_size_bp,if_TTA=False):
        return DataLoader(
            dataset=data_load_ml_fold.TGSDS_TEST(if_TTA=if_TTA,img_size_bp=img_size_bp),
            shuffle=False,
            num_workers=4,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
    with torch.no_grad():
        test_loader = make_loader(batch_size=batch_size,if_TTA = TTA,img_size_bp=img_size_bp)
        pre_y_101_all = []
        for k in KFOLDS:

            weights_list = []
            fold_path = 'fold_experiments/{}/k{}/'.format(ex_name,k)
            # for a in AVE:
            #     w_names = os.listdir(fold_path + 'a{}/weights/'.format(a))
            #     epochs = [os.path.basename(snapshot).split('_')[1] for snapshot in w_names]
            #     epochs = np.array(epochs, dtype=np.int32)
            #     best_epoch = np.max(epochs)
            #     weight_patht = fold_path + 'a{}/weights/PSPNet_{}'.format(a, best_epoch)
            #
            #     weights_list.append(weight_patht)
            for a in AVE:
                w_name = os.listdir(fold_path+'a{}/best_weight_add/'.format(a))[0]
                weight_patht = fold_path+'a{}/best_weight_add/'.format(a)+w_name
                weights_list.append(weight_patht)

            preds_valid_all = []
            for weights_path in weights_list:

                net, starting_epoch = build_network(weights_path, mode_name)
                net.eval()

                test_iterator = tqdm(test_loader)
                pre_y = []
                all_name = []
                if TTA:
                    for vx,vx_f,img_name in test_iterator:
                        vxv,vxv_f = Variable(vx).cuda(), Variable(vx_f).cuda()
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
                            out2_f[t,:,:] = cv2.flip(out2[t,:,:],1)
                        #out2 = cv2.flip(out2,1)

                        outwith = np.stack([out1,out2_f],axis=-1)
                        out = np.mean(outwith,axis=-1)

                        all_name = all_name+img_name
                        pre_y.append(out)


                else:
                    for vx, img_name in test_iterator:
                        vxv = Variable(vx).cuda()
                        out,_,_ = net(vxv)
                        out = F.sigmoid(out)
                        out = torch.squeeze(out.data.cpu(), dim=1)
                        out = out.numpy()


                        all_name = all_name + img_name
                        pre_y.append(out)

                        # print(vy.shape)

                pre_y = np.concatenate(pre_y,axis=0)

                print(pre_y.shape)
                print(len(all_name))


                pre_y_101 = test_unaugment_null(pre_y,img_size_bp)
                preds_valid_all.append(pre_y_101)
            pre_y_101 = np.array(preds_valid_all)
            pre_y_101 = np.transpose(pre_y_101,[1,2,3,0])
            pre_y_101 = np.mean(pre_y_101,axis=-1)#18000,101,101

            pre_y_101_all.append(pre_y_101)
            np.save('m_results/'+mresult_save+'/fold10_{}'.format(k), pre_y_101)

    pre_y_101_all = np.array(pre_y_101_all)
    pre_y_101_all = np.transpose(pre_y_101_all,[1,2,3,0])
    pre_y_101_all = np.mean(pre_y_101_all,axis=-1)#18000,101,101

    if pre_y_101_all.shape[-1] != 101:

        preds_valid2 = np.zeros([pre_y_101_all.shape[0], 101, 101])
        for s in range(pre_y_101_all.shape[0]):


            yt = cv2.resize(pre_y_101_all[s, :, :], dsize=(101, 101))
            # yt = (yt > 0.5).astype(np.float32)
            preds_valid2[s, :, :] = yt

    else:

        preds_valid2 = pre_y_101_all



    preds_valid2 = np.where(preds_valid2>threshold_best,1,0)



    pred_dict = {idx[:-4]: util.RLenc(preds_valid2[i]) for i, idx in enumerate(all_name)}



    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(out_file_name+'.csv')






if __name__ == '__main__':
    m_test()


