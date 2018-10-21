import torch
from torch import nn
from torch.nn import functional as F

from models.common.base_extractors import resnet34
#from models.resnet import resnet34

class HCNET(nn.Module):
    def __init__(self,pretrained):

        super(HCNET, self).__init__()

        self.resnet = resnet34(pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.center = nn.Sequential(
            ConvBn2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )



        self.decoder5 = Decoder(256+512,512,64)
        self.decoder4 = Decoder(64+256,256,64)
        self.decoder3 = Decoder(64+128,128,64)
        self.decoder2 = Decoder(64+64,64,64)
        self.decoder1 = Decoder(64,   32,64)

        self.drop_1 = nn.Dropout2d(p=0.3)
        self.drop_2 = nn.Dropout2d(p=0.5)

        self.logit = nn.Sequential(
            nn.Conv2d(320,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1,padding=0)
        )




    def forward(self,x):

        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]

        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0]
        ],1)

        x = self.conv1(x)  #; print('x',x.size())    #64,64,64
        e2 = self.encoder2(x) #; print('e2',e2.size()) #64,64,64
        e3 = self.encoder3(e2) #; print('e3',e3.size())#128,32,32
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #256,16,16
        e5 = self.encoder5(e4)# ; print('e5',e5.size())#512,8,8

        f = self.center(e5)# ; print('f',f.size()) #256,4,4

        #f= self.drop_1(f)

        d5 = self.decoder5(f,e5)# ; print('d5',d5.size())#64,8,8
        #d5 = self.drop_1(d5)
        d4 = self.decoder4(d5,e4)#; print('d4',d4.size())
        #d4 = self.drop_1(d4)
        d3 = self.decoder3(d4,e3)#; print('d3',d3.size())
        #d3 = self.drop_1(d3)
        d2 = self.decoder2(d3,e2)#; print('d2',d2.size())
        #d2 = self.drop_1(d2)
        d1 = self.decoder1(d2)# ;print('d1',d1.size())
        #d1 = self.drop_1(d1)

        f = torch.cat((
            d1,
            F.upsample(d2,scale_factor=2,mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor=4,mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor=8,mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16,mode='bilinear',align_corners=False)
        ),1)

        f= self.drop_2(f)
        logit = self.logit(f)

        return logit




class Decoder(nn.Module):
    def __init__(self,in_channels,channels,out_channels):
        super(Decoder,self).__init__()

        self.conv1 = ConvBn2d(in_channels,channels,kernel_size=3,padding=1)
        self.conv2 = ConvBn2d(channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()

        self.scse = SCSEBlock(out_channels)

    def forward(self,x,e=None):
        x = F.upsample(x,scale_factor=2,mode='bilinear',align_corners=True)#,align_corners=True

        if e is not None:
            x = torch.cat([x,e],1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.relu(self.conv1(x))
        # x = F.relu(self.conv2(x),inplace=True)
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


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=64):
        super(SCSEBlock, self).__init__()
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
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)

        return torch.add(chn_se, 1, spa_se)

