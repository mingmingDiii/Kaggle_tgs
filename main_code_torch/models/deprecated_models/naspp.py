import torch
from torch import nn
from torch.nn import functional as F

from main_code_torch.models.new_base import senet




class NASPP(nn.Module):
    def __init__(self,pretrained=True,reduction=2):
        super().__init__()
        self.backbone = senet.se_resnet50()

        self.conv1 = self.backbone.layer0

        self.encoder2 = self.backbone.layer1

        self.encoder3 = self.backbone.layer2
        self.encoder4 = self.backbone.layer3

        self.encoder5 = self.backbone.layer4




        # self.scse1 = ModifiedSCSEBlock(64)
        # self.scse2 = ModifiedSCSEBlock(128)
        # self.scse3 = ModifiedSCSEBlock(256)
        # self.scse4 = ModifiedSCSEBlock(512)
        # self.scse5 = ModifiedSCSEBlock(1024)
        # self.scse_center = ModifiedSCSEBlock(256)

        self.center = nn.Sequential(
            ConvBn2d(1024,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )#256,16,16

        self.aspp = ASPP(input_channel=1024,output_channel=1024,depth=256)

        self.decoder5 = Decoder(256+512,512,256,reduction)
        self.decoder4 = Decoder(256+256,256,128,reduction)
        self.decoder3 = Decoder(128,    64,64,reduction)
        #self.decoder2 = Decoder(64,   32,64,reduction)


        # self.center = nn.Sequential(
        #     ConvBn2d(512,512,kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     ConvBn2d(512,256,kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     #nn.MaxPool2d(kernel_size=2,stride=2)
        # )#256,8,8

        self.class_fuse_conv = nn.Sequential(
            ConvBn2d(256,64,kernel_size=1,padding=0),
            nn.ReLU(inplace=True)
        )#64,1,1
        self.class_out = nn.Linear(64,1)

        self.seg_basefuse_conv = nn.Sequential(
            ConvBn2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )#64,128,128


        self.seg_single_logit = nn.Conv2d(64,1,kernel_size=1,padding=0)


        self.fuse_logit = nn.Sequential(
            ConvBn2d(128,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1,padding=0)
        )



    def forward(self, x):

        bts,_,h,w = x.size()

        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]

        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0]
        ],1)

        e1 = self.conv1(x)  # #64,64,64

        e2 = self.encoder2(e1) #256,64,64

        e3 = self.encoder3(e2) #512,32,32

        e4 = self.encoder4(e3)#1024,16,16

        #e5 = self.encoder5(e4) #2048,16,16
        # e5 = self.scse5(e5)

        f = self.aspp(e4)#1024,16,16
        f = self.center(f)#256,16,16


        f_gap = F.adaptive_avg_pool2d(f,output_size=1)#256,1,1
        f_gap = F.dropout(f_gap,p=0.5)
        f_gap_fuse = self.class_fuse_conv(f_gap)#64,1,1

        class_logit = self.class_out(f_gap_fuse.view(bts,64)).view(bts)# 1

        d5 = self.decoder5(f,e3)# ; print('d5',d5.size())#256,32,32

        d4 = self.decoder4(d5,e2)#; print('d4',d4.size())#128,64,64

        seg_base_fuse = self.decoder3(d4)#; print('d3',d3.size())#64,128,128



        # seg_base_fuse = F.upsample(f,size=(h,w),mode='bilinear') #256,128,128
        # seg_base_fuse = self.upsampe_conv(seg_base_fuse) #64,128,128

        seg_base_fuse= F.dropout(seg_base_fuse,p=0.5)#256,128,128

        #seg_base_fuse = hyper#self.seg_basefuse_conv(hyper)#64,128,128

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
    def __init__(self,input_channel=2048,output_channel=1024,depth=256):
        super(ASPP, self).__init__()

        self.conv0 = ConvBn2d_D(input_channel,depth,kernel_size=1,padding=0)
        self.conv1 = ConvBn2d_D(input_channel,depth,kernel_size=3,padding=2,dilation=2)
        self.conv2 = ConvBn2d_D(input_channel,depth,kernel_size=3,padding=4,dilation=4)
        self.conv3 = ConvBn2d_D(input_channel,depth,kernel_size=3,padding=8,dilation=8)

        self.imgl_pool = nn.AdaptiveAvgPool2d(1)
        self.imgl_conv = ConvBn2d_D(input_channel,depth,kernel_size=1,padding=0)

        self.out_conv = ConvBn2d_D(depth*5,output_channel,kernel_size=1,padding=0)


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