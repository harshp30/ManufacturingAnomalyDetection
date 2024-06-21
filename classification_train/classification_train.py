import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from classification_model import create_custom_resnet

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

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Hyperparameters
batch_size = 8
learning_rate = 1e-3
num_epochs = 100
patience = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
train_image_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/classification/train/images'
train_labels_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/classification/train/labels.txt'
val_image_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/classification/val/images'
val_labels_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/classification/val/labels.txt'
model_save_dir = '/home/paperspace/Projects/ManufacturingAnomalyDetection/models/classification'

ensure_dir(model_save_dir)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Datasets and DataLoaders
train_dataset = CustomDataset(train_image_dir, train_labels_path, transforms=transform)
val_dataset = CustomDataset(val_image_dir, val_labels_path, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = 49  # Number of classes

model = create_custom_resnet(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

best_val_loss = float('inf')
early_stopping_counter = 0

def train_model(model, optimizer, num_epochs):
    global best_val_loss, early_stopping_counter
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_loss = validate_model(model, val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            model_path = os.path.join(model_save_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    train_model(model, optimizer, num_epochs)
