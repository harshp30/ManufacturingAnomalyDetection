import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from classification_model import create_custom_resnet
from sklearn.metrics import accuracy_score

class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.images = []
        self.labels = []

        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])
                    self.images.append(filename)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, label

# Hyperparameters
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
test_image_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/classification/test/images'
test_labels_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/classification/test/labels.txt'
model_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/models/classification/model.pth'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
test_dataset = CustomDataset(test_image_dir, test_labels_path, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = 49  # Number of classes

# Load the model
model = create_custom_resnet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate the model
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy on the test set: {accuracy:.4f}')
