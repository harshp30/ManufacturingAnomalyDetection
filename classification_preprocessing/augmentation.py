import os
import cv2
import random
import numpy as np
import albumentations as A

# Paths
stripped_data_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/anomaly_data/classification'
augmented_data_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/augmented_data/classification'

# Ensure directories exist
def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Augmentation pipeline with rotations, flips, color jitter, and blur
augmentation_pipeline = A.Compose([
    A.OneOf([
        A.Rotate(limit=(0, 0), p=1),       # 0 degrees
        A.Rotate(limit=(90, 90), p=1),     # 90 degrees
        A.Rotate(limit=(180, 180), p=1),   # 180 degrees
        A.Rotate(limit=(270, 270), p=1)    # 270 degrees
    ], p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=0.5),
])

# Function to perform augmentation
def augment_data(stripped_data_path, augmented_data_path, num_augmentations=5, seed=42):
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    ensure_dirs([augmented_data_path])

    # List all images
    image_files = [f for f in os.listdir(stripped_data_path) if f.endswith('.png')]
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(stripped_data_path, image_file)
        image = cv2.imread(image_path)

        for i in range(num_augmentations):
            augmented = augmentation_pipeline(image=image)
            aug_image = augmented['image']
            aug_image_filename = image_file.replace('.png', f'_aug_{i}.png')

            cv2.imwrite(os.path.join(augmented_data_path, aug_image_filename), aug_image)

    print(f'Augmentation completed. Augmented data saved to {augmented_data_path}')

# Execute the augmentation
augment_data(stripped_data_path, augmented_data_path)
