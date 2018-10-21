import os
from collections import OrderedDict
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
import click
import numpy as np
import torch
import logging
import sys
from main_code_torch.data_lib import data_load_ml_fold

from main_code_torch.utils import util
from main_code_torch.losses import lovasz_losses
from torch.nn import functional as F
from main_code_torch.data_lib.tgs_agument import *
from tensorboardX import SummaryWriter

import time
import datetime
from main_code_torch.model_include import *
import random
import torch.backends.cudnn as cudnn
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
cudnn.deterministic = True
cudnn.benchmark = True


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

def make_loader(img_list,batch_size, img_size_bp,resize_mode,shuffle=False, transform=None):
    return DataLoader(
        dataset=data_load_ml_fold.TGSDS(img_list=img_list, transform=transform,img_size_bp=img_size_bp,resize_mode=resize_mode),
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=np.random.seed(SEED)
    )




def train_augment(image, mask):

    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)


    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        #c=0
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.125)
        if c == 1:
            image, mask = do_elastic_transform2(image, mask, grid=10,
                                                distort=np.random.uniform(0, 0.1))
        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1,
                                                 angle=np.random.uniform(0, 10))
    return image, mask


def lov_criterion(logit,truth):
    logit = logit.squeeze(1)
    truth = truth.squeeze(1)
    loss = lovasz_losses.lovasz_hinge(logit,truth,per_image=True,ignore=None)
    return loss


def bce_filter(logit,truth,is_filter):
    _,_,h,w = logit.size()
    loss_split = nn.BCEWithLogitsLoss(reduce=False)(logit,truth)
    loss_split = torch.sum(loss_split,dim=[1,2,3])/(h*w)
    loss_split = loss_split*is_filter
    loss_split = torch.sum(loss_split)/(torch.sum(is_filter))
    #print(loss_split.data)

    return loss_split



def test_unaugment_null(prob,img_size_bp=101):
    IMAGE_HEIGHT = img_size_bp
    IMAGE_WIDTH = img_size_bp
    dy0, dy1, dx0, dx1 = compute_center_pad(IMAGE_HEIGHT, IMAGE_WIDTH, factor=32)
    prob = prob[:,dy0:dy0+IMAGE_HEIGHT, dx0:dx0+IMAGE_WIDTH]
    return prob

def val_score_prepare(prob,resize_mode,img_size_bp=101):
    if resize_mode=='pad':
        prob = test_unaugment_null(prob,img_size_bp=img_size_bp)
    else:
        prob = cv2.resize(prob,(img_size_bp,img_size_bp))

    return prob


def mtrain():

    ###### PARAMETERS SETTING ############
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'   #GPU IDS
    ex_name = 'link_dense_fold'                  # experiments name
    model_name = 'link_dense'                    # model name (define in model include)
    describe = " "                               # some descibe for this experiments
    batch_size = 32
    epochs = 800

    lr_decay_step = 15                           # lr decay steps
    early_stop_step = 40                         # early stop steps
    lr_decay_factor = 0.5                        # lr decay facor
    RESIZE_MODE = 'pad'
    lr_decay_mode = 'fix decay'
    opt = 'adam'
    IMAGE_SIZE_BPAD = 101


    KFOLD = [0,1,2,3,4,5,6,7,8,9]                # 10folds
    AVE = [0,1,2,3,4]                            # cycle learning numbers for each folds

    ##################################################### setting experiments dir ###################################################
    exp_dir = 'fold_experiments/{}/'.format(ex_name)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)



    for K in KFOLD:


        ###### make fold dir #######
        fold_path = exp_dir+'k{}/'.format(K)
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)


        train_loader = make_loader(img_list='../data/fold_split2/f{}/train.csv'.format(K),
                                   resize_mode = RESIZE_MODE,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   transform=train_augment,
                                   img_size_bp=IMAGE_SIZE_BPAD)
        valid_loader = make_loader(img_list='../data/fold_split2/f{}/val.csv'.format(K),
                                   resize_mode=RESIZE_MODE,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   transform=None,
                                   img_size_bp=IMAGE_SIZE_BPAD)

        for a in AVE:

            lr = 1e-4

            val_loss = 0.0
            val_miou = 0.0
            val_loss_best = 100000.
            val_miou_best = 0.0
            train_loss_best = 100000.
            train_miou_best = 0.0

            best_valloss_epoch = 0
            best_trainloss_epoch = 0
            early_rec = 0
            lr_rec = 0

            ######  make ave path #######
            ave_path = fold_path+'a{}/'.format(a)
            if not os.path.exists(ave_path):
                os.mkdir(ave_path)

            ### weights path#################
            weights_path = ave_path + 'weights/'
            if not os.path.exists(weights_path):
                os.mkdir(weights_path)
            best_weights_path = ave_path + 'best_weight_add/'
            if not os.path.exists(best_weights_path):
                os.mkdir(best_weights_path)
            ### board path####################
            board_path = ave_path + 'board/'
            if not os.path.exists(board_path):
                os.mkdir(board_path)
            writer = SummaryWriter(board_path)
            ################# config file ###########################################

            config_file = open(ave_path+'config.txt','a')
            config_file.write('Time:\t{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            config_file.write('ex_name:\t{}\n'.format(ex_name))
            config_file.write('model_name:\t{}\n'.format(model_name))
            config_file.write('batch_size:\t{}\n'.format(batch_size))
            config_file.write('IMAGE_SIZE_BPAD:\t{}\n'.format(IMAGE_SIZE_BPAD))
            config_file.write('opt:\t{}\n'.format(opt))
            config_file.write('init_lr:\t{}\n'.format(lr))
            config_file.write('lr_decay_step:\t{}\n'.format(lr_decay_step))
            config_file.write('early_stop_step:\t{}\n'.format(early_stop_step))
            config_file.write('lr_decay_factor:\t{}\n'.format(lr_decay_factor))
            config_file.write('lr_decay_mode:\t{}\n'.format(lr_decay_mode))


            config_file.write('describe:\t{}\n'.format(describe))



            if a==0:
                init_path = None
            else:

                w_names = os.listdir(fold_path + 'a{}/weights/'.format(a-1))
                epochst = [os.path.basename(snapshot).split('_')[1] for snapshot in w_names]
                epochst = np.array(epochst, dtype=np.int32)
                best_epoch = np.max(epochst)
                init_path = fold_path + 'a{}/weights/PSPNet_{}'.format(a-1, best_epoch)

            net, starting_epoch = build_network(init_path, model_name)


            opt_list = {
                'adam': lambda x: optim.Adam(net.parameters(),lr=x),
                'sgd':lambda x: optim.SGD(net.parameters(),lr=x,momentum=0.9, weight_decay=0.0001), # momentum=0.9, weight_decay=0.0001
                'rms':lambda x:optim.RMSprop(net.parameters(),lr=x)
            }
            cri_dict ={
                'lov':lov_criterion,
                'bce_logits':nn.BCEWithLogitsLoss()
            }

            class_criterion = nn.BCEWithLogitsLoss()
            seg_noemp_criterion = bce_filter
            cri = 'lov'
            for epoch in range(starting_epoch, starting_epoch + epochs):

                optimizer = opt_list[opt](lr)
                seg_criterion = cri_dict[cri]

                epoch_losses = []
                epoch_losses_fuse = []
                epoch_losses_seg = []
                epoch_losses_cls = []

                net.train()


                train_iterator = tqdm(train_loader, total=len(train_loader))

                for x, y, is_y in train_iterator:

                    optimizer.zero_grad()

                    x = Variable(x).cuda()
                    if cri == 'lov':
                        y = y.cuda()
                    else:
                        y = Variable(y).cuda()
                    is_y = Variable(is_y).cuda()

                    fuse_logit, seg_logit, class_logit = net(x)
                    seg_loss_fuse = seg_criterion(fuse_logit, y)
                    seg_loss_seg = seg_noemp_criterion(seg_logit, y, is_y)
                    cls_loss = class_criterion(class_logit, is_y)

                    loss = seg_loss_fuse + 0.5 * seg_loss_seg + 0.05 * cls_loss

                    epoch_losses.append(loss.data)

                    epoch_losses_fuse.append(seg_loss_fuse.data)
                    epoch_losses_seg.append(seg_loss_seg.data)
                    epoch_losses_cls.append(cls_loss.data)
                    status = "[{}][{:03d}]" \
                             "all = {:0.5f}," \
                             "fuse = {:0.5f}," \
                             "seg = {:0.5f}," \
                             "cls = {:0.5f}," \
                             "LR = {:0.7f}, " \
                             "vall = {:0.5f}, vmiou = {:0.5f}".format(
                        ex_name, epoch + 1,
                        np.mean(epoch_losses),
                        np.mean(epoch_losses_fuse),
                        np.mean(epoch_losses_seg),
                        np.mean(epoch_losses_cls),
                        lr,
                        val_loss, val_miou)
                    train_iterator.set_description(status)

                    loss.backward()
                    optimizer.step()


                train_loss = np.mean(epoch_losses)

                if train_loss<train_loss_best:
                    train_loss_best = train_loss
                    best_trainloss_epoch = epoch

                with torch.no_grad():
                    # make val
                    net.eval()
                    val_losses = []
                    val_mious = []
                    val_iterator = valid_loader
                    for vx, vy, vy_is in val_iterator:
                        vxv = Variable(vx).cuda()
                        if cri == 'lov':
                            vyv = vy.cuda()
                        else:
                            vyv = Variable(vy).cuda()

                        fuse_logitv, seg_logitv, class_logitv = net(vxv)

                        seg_loss_fusev = seg_criterion(fuse_logitv, vyv)


                        val_losses.append(seg_loss_fusev.data)

                        out_toiou = F.sigmoid(fuse_logitv)
                        out_toiou = torch.squeeze(out_toiou.data.cpu(), dim=1)
                        out_toiou = out_toiou.numpy()
                        vy_toiou = np.squeeze(vy.numpy(),axis=1)


                        out_toiou = val_score_prepare(out_toiou, RESIZE_MODE,img_size_bp=IMAGE_SIZE_BPAD)
                        vy_toiou = val_score_prepare(vy_toiou, RESIZE_MODE,img_size_bp=IMAGE_SIZE_BPAD)

                        miou = util.do_kaggle_metric(predict=out_toiou, truth=vy_toiou)
                        val_mious.append(miou)
                val_loss = np.mean(val_losses)
                val_miou = np.mean(val_mious)

                writer.add_scalar('epoch_train_loss', train_loss, epoch)
                writer.add_scalar('epoch_val_loss', val_loss, epoch)
                writer.add_scalar('epoch_val_miou', val_miou, epoch)
                writer.add_scalar('lr', lr, epoch)


                writer.add_scalars('epoch_group',{
                    'train_loss':train_loss,
                    'val_loss':val_loss,
                    'val_miou':val_miou,
                    'lr':lr
                },epoch)




                # Save best and early stopping
                early_rec += 1
                lr_rec +=1
                if val_miou>=val_miou_best:
                    val_miou_best = val_miou
                    torch.save(net.state_dict(), os.path.join(weights_path, '_'.join(["PSPNet", str(epoch + 1)])))

                    early_rec = 0
                    lr_rec = 0

                if lr_rec>lr_decay_step:
                    print('decay lr',epoch+1)
                    lr = lr *lr_decay_factor
                    lr = max(1e-6,lr)
                    lr_rec=0

                if val_loss<val_loss_best:
                    val_loss_best = val_loss
                    best_valloss_epoch = epoch

                if early_rec>early_stop_step:
                    print('Early stop',epoch+1)
                    print('best val miou',val_miou_best)

                    T_FLAGS = 0
                    torch.save(net.state_dict(), best_weights_path+'best_{}'.format(epoch-early_stop_step))
                    config_file.write('Result:\n')
                    config_file.write('BEST EPOCH:\t{}\n'.format(epoch-early_stop_step))
                    config_file.write('BEST VAL_MIOU:\t{}\n\n'.format(val_miou_best))

                    config_file.write('BEST VALLOSS EPOCH:\t{}\n'.format(best_valloss_epoch))
                    config_file.write('BEST VAL_LOSS:\t{}\n\n'.format(val_loss_best))

                    config_file.write('BEST TRAINLOSS_EPOCH:\t{}\n'.format(best_trainloss_epoch))
                    config_file.write('BEST TRAIN_LOSS:\t{}\n\n\n'.format(train_loss_best))

                    writer.export_scalars_to_json(board_path+'boardlog.json')
                    writer.close()
                    break




if __name__ == '__main__':
    #time.sleep(7200)
    mtrain()


