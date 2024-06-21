import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from singleclass_model import get_model

# Custom Dataset
class LeatherDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Initialize the custom dataset for leather images and masks.

        Parameters:
        image_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        image_transform (callable, optional): Optional transform to be applied on images.
        mask_transform (callable, optional): Optional transform to be applied on masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_files = os.listdir(image_dir)
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        tuple: (image, mask) where image is the transformed image and mask is the corresponding transformed mask.
        """
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace('_image', '_mask'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

# Ensure directory exists
def ensure_dir(directory):
    """
    Ensure that the directory exists. If it does not, create it.

    Parameters:
    directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Dice Loss Function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the Dice loss function.

        Parameters:
        smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Forward pass for Dice loss calculation.

        Parameters:
        y_pred (Tensor): Predicted mask.
        y_true (Tensor): Ground truth mask.

        Returns:
        Tensor: Calculated Dice loss.
        """
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        
        intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
        loss = (2. * intersection + self.smooth) / (y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2) + self.smooth)
        
        return 1 - loss.mean()

# Combined Loss Function
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        """
        Initialize the combined loss function (BCE + Dice).

        Parameters:
        bce_weight (float): Weight for BCE loss in the combined loss.
        """
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, y_pred, y_true):
        """
        Forward pass for combined loss calculation.

        Parameters:
        y_pred (Tensor): Predicted mask.
        y_true (Tensor): Ground truth mask.

        Returns:
        Tensor: Calculated combined loss.
        """
        bce = self.bce_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice

# Hyperparameters
batch_size = 8
learning_rate = 1e-5  # Lowered learning rate
num_epochs = 10
patience = 5  # For early stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
path = ''
image_dir = f'{path}/training_data/transistor/train/images'
mask_dir = f'{path}/training_data/transistor/train/masks'
val_image_dir = f'{path}/training_data/transistor/val/images'
val_mask_dir = f'{path}/training_data/transistor/val/masks'
model_save_dir = f'{path}/models/transistor'

# Ensure the model directory exists
ensure_dir(model_save_dir)

# Transforms
resize_size = (256, 256)

image_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor()
])

# Datasets and Dataloaders
train_dataset = LeatherDataset(image_dir, mask_dir, image_transform, mask_transform)
val_dataset = LeatherDataset(val_image_dir, val_mask_dir, image_transform, mask_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = get_model().to(device)
criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Early stopping parameters
early_stopping_patience = patience
early_stopping_counter = 0
best_val_loss = float('inf')

# Training Loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Train the model.

    Parameters:
    model (nn.Module): The model to train.
    criterion (nn.Module): The loss function.
    optimizer (Optimizer): The optimizer to use.
    scheduler (lr_scheduler): The learning rate scheduler.
    num_epochs (int): The number of epochs to train for.
    """
    global best_val_loss, early_stopping_counter
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Debugging: Print outputs and masks
            if epoch == 0 and images.size(0) > 0:
                print(f"Output sample: {outputs[0].cpu().detach().numpy()}")
                print(f"Mask sample: {masks[0].cpu().detach().numpy()}")

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, criterion, val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            model_path = os.path.join(model_save_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def validate_model(model, criterion, val_loader):
    """
    Validate the model.

    Parameters:
    model (nn.Module): The model to validate.
    criterion (nn.Module): The loss function.
    val_loader (DataLoader): The validation dataloader.

    Returns:
    float: The average validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    return val_loss / len(val_loader.dataset)

if __name__ == "__main__":
    train_model(model, criterion, optimizer, scheduler, num_epochs)
