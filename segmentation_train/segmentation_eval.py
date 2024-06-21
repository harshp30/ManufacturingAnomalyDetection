import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from singleclass_model import get_model
from train import LeatherDataset
import matplotlib.pyplot as plt
import numpy as np

# Paths
test_image_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/transistor/test/images'
test_mask_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/transistor/test/masks'
path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/models/transistor'
model_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/models/transistor/model.pth'

# Hyperparameters
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
resize_size = (256, 256)

image_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor()
])

# Dataset and Dataloader
test_dataset = LeatherDataset(test_image_dir, test_mask_dir, image_transform, mask_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = get_model().to(device)
model.load_state_dict(torch.load(model_path))

# IoU Calculation
def calculate_iou(pred, mask, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = (pred + mask).sum() - intersection
    if union == 0:
        return torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
    iou = intersection / union
    return iou

# Visualize the image, predicted mask, and actual mask
def visualize(image, pred_mask, true_mask, iou, idx):
    image = image.permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    true_mask = true_mask.squeeze().cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow((image * 255).astype(np.uint8))
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title(f'Predicted Mask\nIoU: {iou:.4f}')
    axs[1].axis('off')
    
    axs[2].imshow(true_mask, cmap='gray')
    axs[2].set_title('True Mask')
    axs[2].axis('off')
    
    plt.savefig(f"{path}/visualization_{idx}.png")

# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    total_iou = 0.0
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            for i in range(images.size(0)):
                iou = calculate_iou(outputs[i], masks[i])
                total_iou += iou.item()
                # Visualize predictions
                visualize(images[i], outputs[i], masks[i], iou, idx*batch_size + i)
    
    avg_iou = total_iou / len(test_loader.dataset)
    print(f'Average IoU: {avg_iou:.4f}')

if __name__ == "__main__":
    evaluate_model(model, test_loader)
