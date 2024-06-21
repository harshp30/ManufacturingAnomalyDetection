import os
import cv2
import random
import numpy as np
import albumentations as A

# Paths
path = ''
stripped_data_path = f'{path}/anomaly_data/classification'
augmented_data_path = f'{path}/augmented_data/classification'

def ensure_dirs(paths):
    """
    Ensure that the directories in the given list exist.
    If a directory does not exist, create it.

    Parameters:
    paths (list of str): List of directory paths to check and create if needed.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Augmentation pipeline with rotations, flips, color jitter, and blur
augmentation_pipeline = A.Compose([
    A.OneOf([
        A.Rotate(limit=(0, 0), p=1),       # 0 degrees rotation
        A.Rotate(limit=(90, 90), p=1),     # 90 degrees rotation
        A.Rotate(limit=(180, 180), p=1),   # 180 degrees rotation
        A.Rotate(limit=(270, 270), p=1)    # 270 degrees rotation
    ], p=1),                              # Apply one of the rotations with probability 1
    A.HorizontalFlip(p=0.5),               # Apply horizontal flip with probability 0.5
    A.VerticalFlip(p=0.5),                 # Apply vertical flip with probability 0.5
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # Randomly adjust brightness and contrast
    A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Apply Gaussian blur with a kernel size between 3 and 7
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=0.5), # Randomly crop and resize the image
])

def augment_data(stripped_data_path, augmented_data_path, num_augmentations=5, seed=42):
    """
    Perform data augmentation on images from the stripped_data_path and save the augmented images to augmented_data_path.

    Parameters:
    stripped_data_path (str): Path to the directory containing the original images.
    augmented_data_path (str): Path to the directory where augmented images will be saved.
    num_augmentations (int): Number of augmentations to perform per image. Default is 5.
    seed (int): Random seed for reproducibility. Default is 42.
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    ensure_dirs([augmented_data_path]) # Ensure the augmented data directory exists

    # List all images in the stripped data path
    image_files = [f for f in os.listdir(stripped_data_path) if f.endswith('.png')]
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(stripped_data_path, image_file)
        image = cv2.imread(image_path) # Read the image

        for i in range(num_augmentations): # Perform augmentation num_augmentations times
            augmented = augmentation_pipeline(image=image) # Apply augmentation
            aug_image = augmented['image']
            aug_image_filename = image_file.replace('.png', f'_aug_{i}.png') # Create new filename for augmented image

            cv2.imwrite(os.path.join(augmented_data_path, aug_image_filename), aug_image) # Save augmented image

    print(f'Augmentation completed. Augmented data saved to {augmented_data_path}') # Print completion message

# Execute the augmentation
augment_data(stripped_data_path, augmented_data_path) # Call the function with specified paths
