import torch
from torch import nn
from torch.nn import functional as F

from main_code_torch.models.new_base import dense





class LINK_DENSE(nn.Module):
    def __init__(self,pretrained=True,reduction=2):
        super().__init__()
        self.densenet = dense.densenet121(pretrained=pretrained)
        self.densenet2 = dense.densenet121(pretrained=pretrained)

        self.input_conv = nn.Sequential(
            self.densenet.features.conv0,
            self.densenet.features.norm0,
            self.densenet.features.relu0,
            #self.densenet.features.pool0
        ) #64,64

        self.denseBlocksDown1 = self.densenet.features.denseblock1
        self.transDownBlocks1 = self.densenet.features.transition1

        self.denseBlocksDown2 = self.densenet.features.denseblock2
        self.transDownBlocks2 = self.densenet.features.transition2

        self.denseBlocksDown3 = self.densenet.features.denseblock3
        self.transDownBlocks3 = self.densenet.features.transition3

        self.block = self.densenet.features.denseblock4


        self.transUpBlocks3 = TransitionUp(1024+1024,256,1024)
        self.denseBlocksUp3 = self.densenet2.features.denseblock3

        self.transUpBlocks2 = TransitionUp(1024+512,128,1024)
        self.denseBlocksUp2 = self.densenet2.features.denseblock2

        self.transUpBlocks1 = TransitionUp(512+256,64,512)
        self.denseBlocksUp1 = self.densenet2.features.denseblock1

        self.dout = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0,bias=True)




        self.scse1 = ModifiedSCSEBlock(128)
        self.scse2 = ModifiedSCSEBlock(256)
        self.scse3 = ModifiedSCSEBlock(512)

        self.scse4 = ModifiedSCSEBlock(1024)
        self.scse5 = ModifiedSCSEBlock(512)
        self.scse6 = ModifiedSCSEBlock(256)



        self.center_class = nn.Sequential(
            ConvBn2d(1024,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
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

        x = self.input_conv(x)  #; print('x',x.size())    #64,64,64

        e1 = self.denseBlocksDown1(x) #256,64,64
        e1d = self.transDownBlocks1(e1) #128,32,32
        e1d = self.scse1(e1d)

        e2 = self.denseBlocksDown2(e1d) #512,32,32
        e2d = self.transDownBlocks2(e2) #256,16,16
        e2d = self.scse2(e2d)

        e3 = self.denseBlocksDown3(e2d) #1024,16,16
        e3d = self.transDownBlocks3(e3) #512,8,8
        e3d = self.scse3(e3d)

        center = self.block(e3d) #1024,8,8

        d3u = self.transUpBlocks3(center,e3) #256,16,16
        d3 = self.denseBlocksUp3(d3u)#1024,16,16
        d3 = self.scse4(d3)

        d2u = self.transUpBlocks2(d3,e2) #128,32,32
        d2 = self.denseBlocksUp2(d2u)#512,32,32
        d2 = self.scse5(d2)


        d1u = self.transUpBlocks1(d2,e1) #64,64,64
        d1 = self.denseBlocksUp1(d1u)#256,64,64
        d1 = self.scse6(d1)


        dout = F.upsample(d1,scale_factor=2,mode='bilinear',align_corners=True)#256,64,64



        # class logit
        f = self.center_class(center)# 256,8,8

        f_gap = F.adaptive_avg_pool2d(f,output_size=1)#256,1,1
        f_gap = F.dropout(f_gap,p=0.5)
        f_gap_fuse = self.class_fuse_conv(f_gap)#64,1,1

        class_logit = self.class_out(f_gap_fuse.view(bts,64)).view(bts)# 1


        # seg base
        seg_base_fuse = self.seg_basefuse_conv(dout)#64,128,128


        # seg logit
        seg_logit = self.seg_single_logit(seg_base_fuse)#1,128,128

        # fuse logit
        fuse_feature = torch.cat((
            seg_base_fuse,
            F.upsample(f_gap_fuse,scale_factor=seg_logit.size()[-1],mode='nearest')
        ),1)#128,128,128

        fuse_logit = self.fuse_logit(fuse_feature)#1,128,128

        return fuse_logit,seg_logit,class_logit



class TransitionUp(nn.Module):
    def __init__(self,in_channels,out_channels,up_channels):
        super().__init__()

        self.convTrans = nn.ConvTranspose2d(in_channels=up_channels,out_channels=up_channels,kernel_size=3,stride=2,padding=0,bias=True)

        self.convout = ConvBn2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()

    def forward(self,x,skip=None):

        # x = self.convTrans(x)
        # x = center_crop(x,skip.size(2),skip.size(3))

        x = F.upsample(x,scale_factor=2,mode='bilinear',align_corners=True)
        if skip is not None:
            x = torch.cat([x,skip],1)

        x = self.convout(x)
        x = self.relu(x)

        return x

def center_crop(layer,max_height,max_width):
    _,_,h,w = layer.size()
    xy1 = (w-max_width)//2
    xy2 = (h-max_height)//2

    return layer[:,:,xy2:(xy2+max_height),xy1:(xy1+max_width)]



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
