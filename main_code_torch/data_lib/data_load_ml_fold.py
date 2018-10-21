import numpy as np
from torch.utils import data
import pandas as pd
import os
import glob
import torchvision
import torchvision.transforms as transforms
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
from keras.preprocessing.image import load_img

from main_code_torch.data_lib.tgs_agument import *



class TGSDS(data.Dataset):
    """
    A customized data loader.
    """
    # 83256 20814
    def __init__(self, img_list='../data/',img_size_bp=101,resize_mode='resize',transform=None,if_name = False,if_TTA=False):
        """ Intialize the dataset
        """

        assert resize_mode in ['resize','pad']


        self.img_size_bp = img_size_bp
        self.if_name = if_name
        self.if_TTA = if_TTA
        self.root_path = '../data/'
        self.imgpath = self.root_path+'train/'
        self.transform = transform
        self.resize_mode = resize_mode
        # self.img_transform = Compose([
        # ToTensor(),
        # #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        if self.img_size_bp==101:
            self.img_size_out = 128
        elif self.img_size_bp==202:
            self.img_size_out = 224
        else:
            raise ValueError('wrong img bp size')


        self.img_ids = open(img_list).read().splitlines()


        self.len = len(self.img_ids)





    def __getitem__(self, index):
        """ Get a sample from the dataset
        """

        image = cv2.imread(str(self.imgpath+'images/'+self.img_ids[index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255

        mask = cv2.imread(str(self.imgpath+'masks/'+self.img_ids[index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255

        #image, mask = do_center_pad_to_factor2(image, mask, factor=32)



        if self.transform is not None :
            image, mask = self.transform(image, mask)


        if self.img_size_bp!=101:
            image = cv2.resize(image, dsize=(self.img_size_bp, self.img_size_bp))
            mask = cv2.resize(mask, dsize=(self.img_size_bp, self.img_size_bp))
            mask = (mask > 0.5).astype(np.float32)

        if self.resize_mode=='resize':
            image = cv2.resize(image, dsize=(self.img_size_out, self.img_size_out))
            mask = cv2.resize(mask, dsize=(self.img_size_out, self.img_size_out))
            mask = (mask > 0.5).astype(np.float32)
        else:
            image, mask = do_center_pad_to_factor2(image, mask, factor=32)

        pixel_sum = np.sum(mask)
        if pixel_sum>0:
            is_salt = 1
        else:
            is_salt = 0
        is_salt = np.array(is_salt)

        image_flip = cv2.flip(image,1)

        mask = mask[np.newaxis,...]
        image = image[np.newaxis, ...]
        image_flip = image_flip[np.newaxis,...]

        if self.if_name and self.if_TTA:
            return torch.from_numpy(image).float(),torch.from_numpy(image_flip).float(), torch.from_numpy(mask).float(), str(self.img_ids[index])

        if self.if_name and not self.if_TTA:
            return torch.from_numpy(image).float(), torch.from_numpy(mask).float(),str(self.img_ids[index])

        if not self.if_name and self.if_TTA:
            return torch.from_numpy(image).float(), torch.from_numpy(image_flip).float(), torch.from_numpy(mask).float()
        else:
            return torch.from_numpy(image).float(), torch.from_numpy(mask).float(),torch.from_numpy(is_salt).float()#torch.from_numpy(np.moveaxis(mask, -1, 0)).float()  #self.img_transform(image)


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len



class TGSDS_TEST(data.Dataset):
    """
    A customized data loader.
    """
    # 83256 20814
    def __init__(self,img_size_bp=101,if_TTA=False):
        """ Intialize the dataset
        """


        self.img_size_bp = img_size_bp
        self.if_TTA = if_TTA
        self.root_path = '../data/'
        self.imgpath = self.root_path+'test/'



        self.img_ids = open(self.root_path+'split_list/test.csv').read().splitlines()



        self.len = len(self.img_ids)



    def __getitem__(self, index):
        """ Get a sample from the dataset
        """

        image = cv2.imread(str(self.imgpath + 'images/' + self.img_ids[index]), cv2.IMREAD_GRAYSCALE).astype(
            np.float32) / 255



        if self.img_size_bp!=101:
            image = cv2.resize(image, dsize=(self.img_size_bp, self.img_size_bp))


        image = do_center_pad_to_factor(image, factor=32)
        image_flip = cv2.flip(image,1)
        image = image[np.newaxis, ...]
        image_flip = image_flip[np.newaxis,...]
        if self.if_TTA:
            return torch.from_numpy(image).float(),torch.from_numpy(image_flip).float(),str(self.img_ids[index])
        else:
            return torch.from_numpy(image).float(),str(self.img_ids[index])



    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len






#
if __name__ == '__main__':





    train_transform2 = None
    airimg = TGSDS('train', transform=train_transform2,img_size_bp=101,resize_mode='pad')
    # Use the torch dataloader to iterate through the dataset
    loader = data.DataLoader(airimg, batch_size=24, shuffle=False, num_workers=4)

    # get some images
    dataiter = iter(loader)

    for x in range(100):
        images,masks,is_salt = next(dataiter)
        print(images.shape)
        masks = masks.numpy()
        is_salt = is_salt.numpy()
        print(is_salt[0])
        #plt.figure(figsize=(15,15))
        #plt.subplot(211)
        plt.imshow(images[0,0,:,:])
        #plt.subplot(212)
        plt.imshow(masks[0,0,:,:],alpha=1.0)
        plt.show()






