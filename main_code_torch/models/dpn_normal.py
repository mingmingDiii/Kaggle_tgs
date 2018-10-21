import torch
from torch import nn
from torch.nn import functional as F

from main_code_torch.models.new_base import dpn


class DPN_NORMAL(nn.Module):
    def __init__(self,pretrained=True,reduction=2):
        super().__init__()
        self.basenet = dpn.dpn68b(pretrained='imagenet+5k')

        self.conv1 = self.basenet.features.conv1_1
        self.encoder2 = nn.Sequential(
            self.basenet.features.conv2_1,
            self.basenet.features.conv2_2,
            self.basenet.features.conv2_3,
        ) #64,64

        self.encoder3 = nn.Sequential(
            self.basenet.features.conv3_1,
            self.basenet.features.conv3_2,
            self.basenet.features.conv3_3,
            self.basenet.features.conv3_4,
        )#32,32

        self.encoder4 = nn.Sequential(
            self.basenet.features.conv4_1,
            self.basenet.features.conv4_2,
            self.basenet.features.conv4_3,
            self.basenet.features.conv4_4,
            self.basenet.features.conv4_5,
            self.basenet.features.conv4_6,
            self.basenet.features.conv4_7,
            self.basenet.features.conv4_8,
            self.basenet.features.conv4_9,
            self.basenet.features.conv4_10,
            self.basenet.features.conv4_11,
            self.basenet.features.conv4_12,
        )#16,16

        self.encoder5 = nn.Sequential(
            self.basenet.features.conv5_1,
            self.basenet.features.conv5_2,
            self.basenet.features.conv5_3,
            self.basenet.features.conv5_bn_ac,
        )#8,8


        self.scse1 = ModifiedSCSEBlock(10)
        self.scse2 = ModifiedSCSEBlock(144)
        self.scse3 = ModifiedSCSEBlock(320)
        self.scse4 = ModifiedSCSEBlock(704)
        self.scse5 = ModifiedSCSEBlock(832)
        # self.scse_center = ModifiedSCSEBlock(256)

        self.decoder5 = Decoder(256+704,256,64,reduction)
        self.decoder4 = Decoder(64+320,128,64,reduction)
        self.decoder3 = Decoder(64+144,64,64,reduction)
        self.decoder2 = Decoder(64,   32,64,reduction)


        self.center = nn.Sequential(
            ConvBn2d(832,512,kernel_size=3,padding=1),
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

        e1 = self.conv1(x)  #; print('x',x.size())    #10,64,64
        e1 = self.scse1(e1)
        e2 = self.encoder2(e1) #; print('e2',e2.size()) #[64,80],64,64
        e2 = torch.cat(e2,1)#144
        e2 = self.scse2(e2)
        e3 = self.encoder3(e2) #; print('e3',e3.size())#[128,192],32,32
        e3 = torch.cat(e3, 1)#320
        e3 = self.scse3(e3)
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #[256,448],16,16
        e4 = torch.cat(e4, 1)#704
        e4 = self.scse4(e4)
        e5 = self.encoder5(e4)# ; print('e5',e5.size())#[832],8,8
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

        hyper= F.dropout2d(hyper,p=0.5)#256,128,128

        seg_base_fuse = self.seg_basefuse_conv(hyper)#64,128,128

        seg_logit = self.seg_single_logit(seg_base_fuse)#1,128,128

        fuse_feature = torch.cat((
            seg_base_fuse,
            F.upsample(f_gap_fuse,scale_factor=seg_logit.size()[-1],mode='nearest')
        ),1)#128,128,128

        fuse_logit = self.fuse_logit(fuse_feature)#1,128,128

        return fuse_logit,seg_logit,class_logit



    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

        else:
            raise NotImplementedError


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
