import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # Load the pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        
        # Replace the last fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

def create_custom_resnet(num_classes):
    return CustomResNet(num_classes)
