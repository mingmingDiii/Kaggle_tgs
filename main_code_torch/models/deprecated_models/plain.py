import torch
from torch import nn
from torch.nn import functional as F

from main_code_torch.models.pre_models.common import back_bone_resnet


class PLAIN(nn.Module):
    def __init__(self,pretrained=True,reduction=2):
        super().__init__()
        self.basenet = back_bone_resnet.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.basenet.conv1,
            self.basenet.bn1,
            self.basenet.relu,
            #self.densenet.features.pool0
        ) #64,64

        self.encoder2 = self.basenet.layer1

        self.encoder3 = self.basenet.layer2

        self.encoder4 = self.basenet.layer3

        self.encoder5 = self.basenet.layer4

        # self.encoder5 = nn.Sequential(
        #     self.densenet.features.denseblock4
        # )#16,16


        self.scse1 = ModifiedSCSEBlock(64)
        self.scse2 = ModifiedSCSEBlock(64)
        self.scse3 = ModifiedSCSEBlock(128)
        self.scse4 = ModifiedSCSEBlock(256)
        self.scse5 = ModifiedSCSEBlock(512)
        # self.scse_center = ModifiedSCSEBlock(256)

        self.decoder5 = Decoder(256+256,256,64,reduction)
        self.decoder4 = Decoder(64+128,128,64,reduction)
        self.decoder3 = Decoder(64+64,64,64,reduction)
        self.decoder2 = Decoder(64,   32,64,reduction)


        self.center = nn.Sequential(
            ConvBn2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2,stride=2)
        )#256,8,8

        self.class_fuse_conv = nn.Sequential(
            ConvBn2d(256,64,kernel_size=1,padding=0),
            nn.ReLU(inplace=True)
        )#64,1,1
        self.class_out = nn.Linear(64,1)

        self.seg_basefuse_conv = nn.Sequential(
            ConvBn2d(256,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )#64,128,128

        self.seg_single_logit = nn.Conv2d(64,1,kernel_size=1,padding=0)


        self.fuse_logit = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1,padding=0)
        )



    def forward(self, x):

        bts,_,_,_ = x.size()

        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]

        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0]
        ],1)

        e1 = self.conv1(x)  #; print('x',x.size())    #64,64,64
        e1 = self.scse1(e1)
        e2 = self.encoder2(e1) #; print('e2',e2.size()) #64,64,64
        e2 = self.scse2(e2)
        e3 = self.encoder3(e2) #; print('e3',e3.size())#128,32,32
        e3 = self.scse3(e3)
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #256,16,16
        e4 = self.scse4(e4)
        e5 = self.encoder5(e4)# ; print('e5',e5.size())#512,8,8
        e5 = self.scse5(e5)



        f = self.center(e5)# ; print('f',f.size()) #256,8,8
        #f = self.scse_center(f)

        f_gap = F.adaptive_avg_pool2d(f,output_size=1)#256,1,1
        f_gap = F.dropout(f_gap,p=0.5)
        f_gap_fuse = self.class_fuse_conv(f_gap)#64,1,1

        class_logit = self.class_out(f_gap_fuse.view(bts,64)).view(bts)# 1



        d5 = self.decoder5(f,e4)# ; print('d5',d5.size())#64,16,16
        #d5 = self.drop_1(d5)
        d4 = self.decoder4(d5,e3)#; print('d4',d4.size())#32,32
        #d4 = self.drop_1(d4)
        d3 = self.decoder3(d4,e2)#; print('d3',d3.size())#64,64
        #d3 = self.drop_1(d3)
        d2 = self.decoder2(d3)#; print('d2',d2.size()) #128,128


        hyper = torch.cat((
            d2,
            F.upsample(d3,scale_factor=2,mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor=4,mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=8,mode='bilinear',align_corners=False)
        ),1) #256,128,128

        hyper= F.dropout(hyper,p=0.5)#256,128,128

        seg_base_fuse = self.seg_basefuse_conv(hyper)#64,128,128

        seg_logit = self.seg_single_logit(seg_base_fuse)#1,128,128

        fuse_feature = torch.cat((
            seg_base_fuse,
            F.upsample(f_gap_fuse,scale_factor=seg_logit.size()[-1],mode='nearest')
        ),1)#128,128,128

        fuse_logit = self.fuse_logit(fuse_feature)#1,128,128

        return fuse_logit,seg_logit,class_logit




class Decoder(nn.Module):
    def __init__(self,in_channels,channels,out_channels,reduction=2):
        super(Decoder,self).__init__()

        self.conv1 = ConvBn2d(in_channels,channels,kernel_size=3,padding=1)
        self.conv2 = ConvBn2d(channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()

        self.scse = ModifiedSCSEBlock(out_channels,reduction)

    def forward(self,x,e=None):
        x = F.upsample(x,scale_factor=2,mode='bilinear',align_corners=True)#,align_corners=True

        if e is not None:
            x = torch.cat([x,e],1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.scse(x)


        return x






class ConvBn2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(ConvBn2d, self).__init__()

        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        return self.convbn(x)

class ConvBn2d_D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,dilation=1):
        super(ConvBn2d_D, self).__init__()

        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.convbn(x)


class ModifiedSCSEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ModifiedSCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)

        spa_se = self.spatial_se(x)
        return torch.mul(torch.mul(x, chn_se), spa_se)



class ASPP(nn.Module):
    def __init__(self,depth=256):
        super(ASPP, self).__init__()

        self.conv0 = ConvBn2d_D(1024,depth,kernel_size=1,padding=0)
        self.conv1 = ConvBn2d_D(1024,depth,kernel_size=3,padding=2,dilation=2)
        self.conv2 = ConvBn2d_D(1024,depth,kernel_size=3,padding=4,dilation=4)
        self.conv3 = ConvBn2d_D(1024,depth,kernel_size=3,padding=8,dilation=8)

        self.imgl_pool = nn.AdaptiveAvgPool2d(1)
        self.imgl_conv = ConvBn2d_D(1024,depth,kernel_size=1,padding=0)

        self.out_conv = ConvBn2d_D(depth*5,1024,kernel_size=1,padding=0)


    def forward(self, x):


        h, w = x.size(2), x.size(3)


        conv_1x1 = self.conv0(x)
        conv_3x3_1 = self.conv1(x)
        conv_3x3_2 = self.conv2(x)
        conv_3x3_3 = self.conv3(x)

        img_level = self.imgl_pool(x)
        img_level = self.imgl_conv(img_level)
        img_level = F.upsample(img_level,size=(h,w),mode='bilinear')

        out = torch.cat([conv_1x1,conv_3x3_1,conv_3x3_2,conv_3x3_3,img_level],1)
        out = self.out_conv(out)

        return out