import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the CustomResNet model.

        Parameters:
        num_classes (int): Number of output classes.
        """
        super(CustomResNet, self).__init__()
        # Load the pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        
        # Replace the last fully connected layer to match the number of output classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        x (Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
        Tensor: Output tensor with logits for each class.
        """
        return self.model(x)

def create_custom_resnet(num_classes):
    """
    Function to create an instance of CustomResNet.

    Parameters:
    num_classes (int): Number of output classes.

    Returns:
    CustomResNet: An instance of the CustomResNet model.
    """
    return CustomResNet(num_classes)
