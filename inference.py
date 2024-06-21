import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from classification_model import CustomResNet
from segmentation_model import ResNetUNet  # Ensure this is the correct import for your segmentation model

# Define the class labels
class_labels = {
    'grid_bent': 0,
    'grid_broken': 1,
    'grid_glue': 2,
    'grid_good': 3,
    'grid_metal_contimation': 4,
    'grid_thread': 5,
    'leather_color': 6,
    'leather_cut': 7,
    'leather_fold': 8,
    'leather_glue': 9,
    'leather_good': 10,
    'leather_poke': 11,
    'metalnut_bent': 12,
    'metalnut_color': 13,
    'metalnut_flip': 14,
    'metalnut_good': 15,
    'metalnut_scratch': 16,
    'screw_good': 17,
    'screw_manipulated_front': 18,
    'screw_scratch_head': 19,
    'screw_scratch_neck': 20,
    'screw_scratch_side': 21,
    'screw_thread_top': 22,
    'cable_bent_wire': 23,
    'cable_cable_swap': 24,
    'cable_combined': 25,
    'cable_cut_inner_insulation': 26,
    'cable_cut_outer_insulation': 27,
    'cable_good': 28,
    'cable_missing_cable': 29,
    'cable_missing_wire': 30,
    'cable_poke_insulation': 31,
    'tile_crack': 32,
    'tile_glue_strip': 33,
    'tile_good': 34,
    'tile_gray_stroke': 35,
    'tile_oil': 36,
    'tile_rough': 37,
    'transistor_bent_lead': 38,
    'transistor_cut_lead': 39,
    'transistor_damaged_case': 40,
    'transistor_good': 41,
    'transistor_misplaced': 42,
    'wood_color': 43,
    'wood_combined': 44,
    'wood_good': 45,
    'wood_hole': 46,
    'wood_liquid': 47,
    'wood_scratch': 48
}

# Reverse the class_labels dictionary for easier lookup
class_labels_rev = {v: k for k, v in class_labels.items()}

# Custom Dataset for Inference
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, img_path

# Ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to get the segmentation model
def get_segmentation_model():
    return ResNetUNet(n_classes=1)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
image_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/cable/test/images/test_image_0.png'
segmentation_models_base_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/models'
output_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/output'
classification_model_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/models/classification/model.pth'

ensure_dir(output_dir)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = InferenceDataset([image_path], transforms=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the trained classification model
num_classes = 49
classification_model = CustomResNet(num_classes=num_classes).to(device)
classification_model.load_state_dict(torch.load(classification_model_path))
classification_model.eval()

# Function to save results
def save_results(image, pred_mask, classification, idx):
    image = image.permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title(f'Predicted Mask\nClass: {classification}')
    axs[1].axis('off')

    plt.savefig(os.path.join(output_dir, f'result.png'))
    plt.close()

# Inference function
def inference(dataloader):
    with torch.no_grad():
        for idx, (images, filenames) in enumerate(dataloader):
            images = [image.to(device) for image in images]
            
            # Classify the image
            classification_outputs = classification_model(images[0].unsqueeze(0))
            _, preds = torch.max(classification_outputs, 1)

            for i, image in enumerate(images):
                class_idx = preds[i].item()
                class_name = class_labels_rev[class_idx]
                category_name = class_name.split('_')[0]
                
                # Path to the segmentation model for the predicted class
                segmentation_model_path = os.path.join(segmentation_models_base_path, category_name, 'model.pth')

                if not os.path.exists(segmentation_model_path):
                    print(f"No segmentation model found for class {class_name}. Skipping...")
                    continue

                # Load the appropriate segmentation model
                segmentation_model = get_segmentation_model().to(device)
                segmentation_model.load_state_dict(torch.load(segmentation_model_path))
                segmentation_model.eval()

                # Generate the segmentation mask
                segmentation_output = segmentation_model(image.unsqueeze(0))
                pred_mask = segmentation_output.squeeze(0)
                
                save_results(image, pred_mask, class_name, idx)
                print(f"Processed image {filenames[i]}, Class: {class_name}")

if __name__ == "__main__":
    inference(dataloader)
