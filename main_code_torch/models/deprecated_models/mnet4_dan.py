import torch
from torch import nn
from torch.nn import functional as F

from models.common import mextractors2




class MNET4_DAN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.densenet = mextractors2.densenet121(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.densenet.features.conv0,
            self.densenet.features.norm0,
            self.densenet.features.relu0,
            #self.densenet.features.pool0
        ) #64,64

        self.encoder2 = nn.Sequential(
            self.densenet.features.denseblock1,
            self.densenet.features.transition1
        ) #32,32

        self.encoder3 = nn.Sequential(
            self.densenet.features.denseblock2,
            self.densenet.features.transition2
        )#16,16

        self.encoder4 = nn.Sequential(
            self.densenet.features.denseblock3,
            self.densenet.features.transition3
        )#16,16

        self.encoder5 = nn.Sequential(
            self.densenet.features.denseblock4
        )#16,16


        #self.aspp = ASPP(depth=256)#16,16


        self.dan = DANetHead(in_channels=1024,out_channels=256)


        self.center = nn.Sequential(
            # ConvBn2d(1024,512,kernel_size=3,padding=1),
            # nn.ReLU(inplace=True),
            ConvBn2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#8,8



        self.decoder5 = Decoder(256+256,256,64)
        self.decoder4 = Decoder(64+128,128,64)
        self.decoder3 = Decoder(64+64,64,64)
        self.decoder2 = Decoder(64,   32,64)

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
        e2 = self.encoder2(e1) #; print('e2',e2.size()) #128,32,32
        e3 = self.encoder3(e2) #; print('e3',e3.size())#256,16,16
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #512,16,16
        e5 = self.encoder5(e4)# ; print('e5',e5.size())#1024,16,16


        e5 = self.dan(e5)

        f = self.center(e5)# ; print('f',f.size()) #256,8,8

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




class Decoder(nn.Module):
    def __init__(self,in_channels,channels,out_channels):
        super(Decoder,self).__init__()

        self.conv1 = ConvBn2d(in_channels,channels,kernel_size=3,padding=1)
        self.conv2 = ConvBn2d(channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()

        #self.scse = SCSEBlock(out_channels)

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
        #x = self.scse(x)


        return x



class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        x = self.conv8(feat_sum)

        #@output = [x]
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





class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self,in_dim):
        super(PAM_Module,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height)
        energy = torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(proj_value,attention.permute(0, 2, 1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self,in_dim):
        super(CAM_Module,self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize,C,-1)
        proj_key = x.view(m_batchsize,C,-1).permute(0,2,1)
        energy = torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize,C,-1)

        out = torch.bmm(attention,proj_value)
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

