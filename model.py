"""U-Net contracting path -> clinical mixture of image experts -> adapted ResNet50."""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class GuidedClassifier(nn.Module):
    def __init__(self, clinical_dim, pretrained=True):
        super().__init__()
        self.encoder=nn.Sequential(DoubleConv(3,16),nn.MaxPool2d(2),DoubleConv(16,32),nn.MaxPool2d(2),
                                   DoubleConv(32,64),nn.MaxPool2d(2),DoubleConv(64,128))
        self.experts=nn.ModuleList([nn.Sequential(nn.Conv2d(128,64,3,padding=d,dilation=d,bias=False),
                                    nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.Conv2d(64,128,1)) for d in (1,2,3)])
        self.router=nn.Sequential(nn.Linear(clinical_dim,32),nn.ReLU(),nn.Dropout(0.15),nn.Linear(32,3))
        # Begin with nearly uniform routing. Experts are NOT assigned disease labels.
        nn.init.normal_(self.router[-1].weight,std=0.01)
        nn.init.zeros_(self.router[-1].bias)
        self.adapter=nn.Sequential(nn.Conv2d(128,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.resnet=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # U-Net already downsamples 224->28: replace the RGB stem and remove its pooling.
        # All 16 pretrained bottleneck blocks (3,4,6,3) are retained.
        self.resnet.conv1=nn.Identity()
        self.resnet.bn1=nn.Identity()
        self.resnet.relu=nn.Identity()
        self.resnet.maxpool=nn.Identity()
        self.resnet.fc=nn.Sequential(nn.Dropout(0.3),nn.Linear(2048,3))

    def forward(self, images, clinical, uniform=False):
        shared=self.encoder(images)
        weights=torch.softmax(self.router(clinical),dim=1)
        if uniform: weights=torch.ones_like(weights)/3
        guided=sum(expert(shared)*weights[:,i,None,None,None] for i,expert in enumerate(self.experts))
        return self.resnet(self.adapter(guided)),weights

    def freeze_residual_backbone(self, frozen):
        for name,p in self.resnet.named_parameters():
            if not name.startswith('fc.'): p.requires_grad_(not frozen)

    def fix_pretrained_bn(self):
        # Small batches and feature-domain shift: retain ImageNet running statistics.
        for module in self.resnet.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
