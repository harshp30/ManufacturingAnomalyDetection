# singleclass_model.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        """
        Initialize the ResNet-UNet model.

        Parameters:
        n_classes (int): Number of output classes for the segmentation task.
        """
        super(ResNetUNet, self).__init__()
        self.base_model = models.resnet34(pretrained=True)  # Load pre-trained ResNet-34 model
        self.base_layers = list(self.base_model.children())  # Extract the layers of ResNet-34
        
        # Define the layers from ResNet-34 to be used in the encoder part of UNet
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # First few layers, output shape: 64 x 128 x 128
        self.layer0_1x1 = nn.Conv2d(64, 64, kernel_size=1)  # 1x1 convolution to adjust channel size
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # Next layers, output shape: 64 x 64 x 64
        self.layer1_1x1 = nn.Conv2d(64, 64, kernel_size=1)  # 1x1 convolution
        self.layer2 = self.base_layers[5]  # Output shape: 128 x 32 x 32
        self.layer2_1x1 = nn.Conv2d(128, 128, kernel_size=1)  # 1x1 convolution
        self.layer3 = self.base_layers[6]  # Output shape: 256 x 16 x 16
        self.layer3_1x1 = nn.Conv2d(256, 256, kernel_size=1)  # 1x1 convolution
        self.layer4 = self.base_layers[7]  # Output shape: 512 x 8 x 8
        self.layer4_1x1 = nn.Conv2d(512, 512, kernel_size=1)  # 1x1 convolution
        
        # Define the upsampling layers for the decoder part of UNet
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling layer
        self.conv_up3 = self.double_conv(256 + 512, 256)  # Double convolution after concatenating
        self.conv_up2 = self.double_conv(128 + 256, 128)
        self.conv_up1 = self.double_conv(64 + 128, 64)
        self.conv_up0 = self.double_conv(64 + 64, 64)
        
        # Convolution layers for initial and final processing
        self.conv_original_size0 = self.double_conv(3, 64)
        self.conv_original_size1 = self.double_conv(64, 64)
        self.conv_original_size2 = self.double_conv(64 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)  # Final 1x1 convolution to get the output

    def double_conv(self, in_channels, out_channels):
        """
        Define a double convolution block with ReLU activation.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

        Returns:
        nn.Sequential: Double convolution block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        """
        Define the forward pass of the model.

        Parameters:
        input (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after segmentation.
        """
        x_original = self.conv_original_size0(input)  # Initial double convolution
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)  # Pass through initial layers
        layer1 = self.layer1(layer0)  # Pass through first block of ResNet
        layer2 = self.layer2(layer1)  # Pass through second block of ResNet
        layer3 = self.layer3(layer2)  # Pass through third block of ResNet
        layer4 = self.layer4(layer3)  # Pass through fourth block of ResNet
        
        layer4 = self.layer4_1x1(layer4)  # Adjust channels with 1x1 convolution
        x = self.upsample(layer4)  # Upsample
        layer3 = self.layer3_1x1(layer3)  # Adjust channels
        x = torch.cat([x, layer3], dim=1)  # Concatenate with corresponding encoder layer
        x = self.conv_up3(x)  # Double convolution
        
        x = self.upsample(x)  # Repeat for remaining layers
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
        
        out = self.conv_last(x)  # Final convolution to get the output
        return torch.sigmoid(out)  # Apply sigmoid activation

def get_model(n_classes=1):
    """
    Get the ResNet-UNet model.

    Parameters:
    n_classes (int): Number of output classes for the segmentation task.

    Returns:
    ResNetUNet: Instance of the ResNet-UNet model.
    """
    model = ResNetUNet(n_classes)
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)
