import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetUNet(nn.Module):
    
    def __init__(self, num_classes=2):
        super(ResNetUNet, self).__init__()
        resnet = models.resnet50(pretrained=True)

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.input_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64→256
        self.encoder2 = resnet.layer2  # 256→512
        self.encoder3 = resnet.layer3  # 512→1024
        self.bottleneck = resnet.layer4  # 1024→2048

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.input_block(x)
        x1 = self.maxpool(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.bottleneck(x4)

        d1 = F.relu(self.up1(x5) + x4)
        d1 = self.conv1(d1)

        d2 = F.relu(self.up2(d1) + x3)
        d2 = self.conv2(d2)

        d3 = F.relu(self.up3(d2) + x2)
        d3 = self.conv3(d3)

        d4 = F.relu(self.up4(d3) + x0)
        d4 = self.conv4(d4)

        d5 = self.up5(d4)
        out = self.out_conv(d5)
        return out

