import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from models.common import mextractors2


InPlaceABNSync = nn.BatchNorm2d
torch_ver = torch.__version__[:3]

class DENSE_OCNET_H_S(nn.Module):
    def __init__(self,pretrained=True):
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


        self.scse1 = ModifiedSCSEBlock(64)
        self.scse2 = ModifiedSCSEBlock(128)
        self.scse3 = ModifiedSCSEBlock(256)
        self.scse4 = ModifiedSCSEBlock(512)
        self.scse5 = ModifiedSCSEBlock(1024)
        #
        # self.scse_center = ModifiedSCSEBlock(256)



        #self.psp = PSPModule(psp_size, 1024, sizes)#16,16

        self.context = nn.Sequential(
                ASP_OC_Module(1024, 512),
                )

        # self.center = nn.Sequential(
        #     ConvBn2d(1024,512,kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     ConvBn2d(512,256,kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2,stride=2)
        # )#8,8
        #
        #
        #
        self.decoder5 = Decoder(512+128,128,64)
        self.decoder4 = Decoder(64+64,64,64)
        self.decoder3 = Decoder(64,32,64)

        #
        self.drop_out2 = nn.Dropout2d(p=0.5)

        self.logit = nn.Sequential(
            nn.Conv2d(64*3,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1,padding=0)
        )


        #self.cls = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):


        mean = [0.485,0.456,0.406]
        std = [0.229,0.224,0.225]

        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0]
        ],1)

        bahs, chs, h, w = x.size()
        e1 = self.conv1(x)  #; print('x',x.size())    #64,64,64
        e1 = self.scse1(e1)
        e2 = self.encoder2(e1) #; print('e2',e2.size()) #128,32,32
        e2 = self.scse2(e2)
        e3 = self.encoder3(e2) #; print('e3',e3.size())#256,16,16
        e3 = self.scse3(e3)
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #512,16,16
        e4 = self.scse4(e4)
        e5 = self.encoder5(e4)# ; print('e5',e5.size())#1024,16,16
        e5 = self.scse5(e5)

        #e5 = self.drop_out2(e5)
        #f = self.center(e5)# ; print('f',f.size()) #256,8,8
        f = self.context(e5) # 512,16,16


        d5 = self.decoder5(f,e2)# ; print('d5',d5.size())#64,32,32
        #d5 = self.drop_1(d5)
        d4 = self.decoder4(d5,e1)#; print('d4',d4.size())#64,64
        #d4 = self.drop_1(d4)
        d3 = self.decoder3(d4)#; print('d3',d3.size())#128,128
        #d3 = self.drop_1(d3)



        f = torch.cat((
            d3,
            F.upsample(d4,scale_factor=2,mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=4,mode='bilinear',align_corners=False)
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


class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     InPlaceABNSync(out_features),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                           key_channels=out_features // 2, value_channels=out_features,
                                                           dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(out_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(out_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            InPlaceABNSync(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            if torch_ver == '0.4':
                context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
            elif torch_ver == '0.3':
                context = F.upsample(input=context, size=(h, w), mode='bilinear')
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            InPlaceABNSync(out_channels),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


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
