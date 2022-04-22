import torch
from torch import nn
import torch.nn.functional as F
from .resnet import *
from .mobilenet import *


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, args=None):
        super(PSPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = args.use_ppm
        self.use_aspp = args.use_aspp
        self.criterion = criterion
        self.args = args

        ### [requires implementation] parameter/module definitions
        '''
        You are required to reproduce PPM of PSPNet. Also, the you should also implement the ASPP module used in DeepLab-V3+
        PSPNet: https://jiaya.me/papers/PSPNet_cvpr17.pdf
        DeepLab-V3+: https://arxiv.org/abs/1802.02611
        If you correctly implement both PPM and ASPP module, they should achieve close performance (gap <= 2%). 
        
        Dilations should be added to the model to make sure the output feature map is roughly 1/8 of the input feature.
            Example: if the input spatial size is 473x473 and the output feature map's spatial size should be 60x60.
        
        For resnet-based backbones, you can reuse the pretrained layer like:
        self.layer0 = nn.Sequential(network.conv1, network.bn1, network.relu, network.conv2, network.bn2, network.relu, network.conv3, network.bn3, network.relu, network.maxpool)

        For mobilenet-based backbone, you can reuse the pretrained layer like:
        self.layer0 = nn.Sequential(*[network.features[_i] for _i in [xxx]])
        '''

        if self.use_ppm:
            self.enrich_module = PPM()
        elif self.use_aspp:
            self.enrich_module = ASPP()
        elif self.use_ocr:
            self.enrich_module = OCR()

    def forward(self, x, y=None, preact=False):

        ### [requires implementation] forward and get predicted logits

        if self.training:
            ### [requires implementation] calculate losses for training

            return x.max(1)[1], main_loss, aux_loss
        else:
            return x
