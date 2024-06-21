# singleclass_model.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super(ResNetUNet, self).__init__()
        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # 64, 128, 128
        self.layer0_1x1 = nn.Conv2d(64, 64, kernel_size=1)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # 64, 64, 64
        self.layer1_1x1 = nn.Conv2d(64, 64, kernel_size=1)
        self.layer2 = self.base_layers[5]  # 128, 32, 32
        self.layer2_1x1 = nn.Conv2d(128, 128, kernel_size=1)
        self.layer3 = self.base_layers[6]  # 256, 16, 16
        self.layer3_1x1 = nn.Conv2d(256, 256, kernel_size=1)
        self.layer4 = self.base_layers[7]  # 512, 8, 8
        self.layer4_1x1 = nn.Conv2d(512, 512, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = self.double_conv(256 + 512, 256)
        self.conv_up2 = self.double_conv(128 + 256, 128)
        self.conv_up1 = self.double_conv(64 + 128, 64)
        self.conv_up0 = self.double_conv(64 + 64, 64)
        
        self.conv_original_size0 = self.double_conv(3, 64)
        self.conv_original_size1 = self.double_conv(64, 64)
        self.conv_original_size2 = self.double_conv(64 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        
        out = self.conv_last(x)
        
        return torch.sigmoid(out)

def get_model(n_classes=1):
    model = ResNetUNet(n_classes)
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)
