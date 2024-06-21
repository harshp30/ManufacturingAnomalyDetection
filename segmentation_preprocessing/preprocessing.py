import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Paths
stripped_data_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/augmented_data/transistor'
processed_data_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/training_data/transistor'

# Ensure directories exist
def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to resize images and masks to a consistent size
def resize_image_and_mask(image, mask, size=(256, 256)):
    resized_image = cv2.resize(image, size)
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask

# Function to split data into training, validation, and testing sets with oversampling
def split_data(stripped_data_path, processed_data_path):
    # Categories
    categories = ['good', 'bent_lead', 'cut_lead', 'damaged_case', 'misplaced']
    
    # Ensure training, validation, and testing directories exist
    ensure_dirs([
        os.path.join(processed_data_path, 'train', 'images'),
        os.path.join(processed_data_path, 'train', 'masks'),
        os.path.join(processed_data_path, 'val', 'images'),
        os.path.join(processed_data_path, 'val', 'masks'),
        os.path.join(processed_data_path, 'test', 'images'),
        os.path.join(processed_data_path, 'test', 'masks')
    ])
    
    all_images = []
    all_masks = []
    labels = []
    
    for category in categories:
        image_files = [f for f in os.listdir(stripped_data_path) if f.startswith(category) and f.endswith('_image.png')]
        
        # Debugging: Print the number of images found for each category
        print(f'Found {len(image_files)} images for category: {category}')
        
        for image_file in image_files:
            image_path = os.path.join(stripped_data_path, image_file)
            mask_path = image_path.replace('_image.png', '_mask.png')
            
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Resize images and masks to a consistent size
            resized_image, resized_mask = resize_image_and_mask(image, mask)
            
            all_images.append(resized_image)
            all_masks.append(resized_mask)
            labels.append(category)
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_masks = np.array(all_masks)
    labels = np.array(labels)
    
    # Ensure there are images for each category
    for category in categories:
        if category not in labels:
            raise ValueError(f"No images found for category: {category}")
    
    # Oversample the minority class
    max_count = max(np.bincount(labels == 'good'))
    oversampled_images = []
    oversampled_masks = []
    
    for category in categories:
        category_indices = np.where(labels == category)[0]
        if len(category_indices) == 0:
            raise ValueError(f"No images found for category: {category}")
        
        oversample_indices = np.random.choice(category_indices, max_count, replace=True)
        oversampled_images.append(all_images[oversample_indices])
        oversampled_masks.append(all_masks[oversample_indices])
    
    oversampled_images = np.concatenate(oversampled_images)
    oversampled_masks = np.concatenate(oversampled_masks)
    
    # Split into training, validation, and testing sets (60% train, 20% val, 20% test)
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        oversampled_images, oversampled_masks, test_size=0.4, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=0.5, random_state=42)
    
    # Save images and masks
    def save_images_and_masks(images, masks, set_name):
        for i in range(len(images)):
            image_path = os.path.join(processed_data_path, set_name, 'images', f'{set_name}_image_{i}.png')
            mask_path = os.path.join(processed_data_path, set_name, 'masks', f'{set_name}_mask_{i}.png')
            cv2.imwrite(image_path, images[i])
            cv2.imwrite(mask_path, masks[i])
    
    save_images_and_masks(train_images, train_masks, 'train')
    save_images_and_masks(val_images, val_masks, 'val')
    save_images_and_masks(test_images, test_masks, 'test')

    print(f'Data split into training, validation, and testing sets and saved to {processed_data_path}.')

# Execute the data splitting
split_data(stripped_data_path, processed_data_path)
