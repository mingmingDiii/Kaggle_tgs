import torch
from torch import nn
from torch.nn import functional as F

from models.common import mextractors4
#from models.common import resnet

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class MPSPNet2_SCSE_RE4_RES34(nn.Module):
    def __init__(self, pretrained=True,reduction=2):
        super().__init__()
        #self.densenet = mextractors2.densenet121(pretrained=pretrained)
        self.resnet = mextractors4.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.densenet.features.pool0
        ) #64,64

        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2

        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4


        self.scse1 = ModifiedSCSEBlock(64)
        self.scse2 = ModifiedSCSEBlock(64)
        self.scse3 = ModifiedSCSEBlock(128)
        self.scse4 = ModifiedSCSEBlock(256)
        self.scse5 = ModifiedSCSEBlock(512)

        self.scse_center = ModifiedSCSEBlock(256)



        #self.psp = PSPModule(psp_size, 1024, sizes)#16,16



        self.center = nn.Sequential(
            ConvBn2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#8,8



        self.decoder5 = Decoder(256+128,128,64,reduction)
        self.decoder4 = Decoder(64+64,64,64,reduction)
        self.decoder3 = Decoder(64+64,64,64,reduction)
        self.decoder2 = Decoder(64,   32,64,reduction)

        self.drop_out2 = nn.Dropout2d(p=0.5)

        self.logit = nn.Sequential(
            nn.Conv2d(256,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1,padding=0)
        )


    def forward(self, x):

        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]

        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0]
        ],1)

        e1 = self.conv1(x)  #; print('x',x.size())    #64,64,64
        e1 = self.scse1(e1)
        e2 = self.encoder2(e1) #; print('e2',e2.size()) #64,32,32
        e2 = self.scse2(e2)
        e3 = self.encoder3(e2) #; print('e3',e3.size())#128,16,16
        e3 = self.scse3(e3)
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #256,16,16
        e4 = self.scse4(e4)
        e5 = self.encoder5(e4)# ; print('e5',e5.size())#512,16,16
        e5 = self.scse5(e5)

        f = self.center(e5)# ; print('f',f.size()) #256,8,8
        f = self.scse_center(f)

        #f= self.drop_1(f)

        d5 = self.decoder5(f,e3)# ; print('d5',d5.size())#64,16,16
        #d5 = self.drop_1(d5)
        d4 = self.decoder4(d5,e2)#; print('d4',d4.size())#32,32
        #d4 = self.drop_1(d4)
        d3 = self.decoder3(d4,e1)#; print('d3',d3.size())#64,64
        #d3 = self.drop_1(d3)
        d2 = self.decoder2(d3)#; print('d2',d2.size()) #128,128


        f = torch.cat((
            d2,
            F.upsample(d3,scale_factor=2,mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor=4,mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=8,mode='bilinear',align_corners=False)
        ),1)

        f= self.drop_out2(f)

        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.logit(f)#, self.classifier(auxiliary)

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
