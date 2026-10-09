"""Preserve ImageNet image features while adding clinically routed U-Net features."""
import torch
from torch import nn
from torchvision.models import resnet50,ResNet50_Weights
from model import DoubleConv

class ImprovedClassifier(nn.Module):
    def __init__(self,clinical_dim,pretrained=True):
        super().__init__()
        self.encoder=nn.Sequential(DoubleConv(3,16),nn.MaxPool2d(2),DoubleConv(16,32),nn.MaxPool2d(2),DoubleConv(32,64),nn.MaxPool2d(2),DoubleConv(64,128))
        self.experts=nn.ModuleList([nn.Sequential(nn.Conv2d(128,64,3,padding=d,dilation=d,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,512,1)) for d in (1,2,3)])
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight,std=.001);nn.init.zeros_(expert[-1].bias)
        self.router=nn.Sequential(nn.Linear(clinical_dim,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,3))
        self.resnet=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.resnet.fc=nn.Sequential(nn.Dropout(.3),nn.Linear(2048,3))

    def forward(self,x,clinical,uniform=False):
        shared=self.encoder(x)
        w=self.router(clinical).softmax(1)
        if uniform:w=torch.ones_like(w)/3
        guided=sum(expert(shared)*w[:,i,None,None,None] for i,expert in enumerate(self.experts))
        r=self.resnet
        image=r.maxpool(r.relu(r.bn1(r.conv1(x))))
        image=r.layer2(r.layer1(image))
        # Both tensors are B x 512 x 28 x 28 at 224px input.
        fused=image+guided
        features=r.layer4(r.layer3(fused))
        return r.fc(r.avgpool(features).flatten(1)),w

    def freeze_backbone(self,frozen):
        for name,p in self.resnet.named_parameters():
            if not name.startswith('fc.'):p.requires_grad_(not frozen)

    def fix_bn(self):
        for module in self.resnet.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
