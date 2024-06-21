import os
import cv2
import random
import numpy as np
import albumentations as A

# Paths
path = ''
stripped_data_path = f'{path}/anomaly_data/classification'
augmented_data_path = f'{path}/augmented_data/classification'

# Ensure directories exist
def ensure_dirs(paths):
    """
    Ensure that the directories in the given list exist. If they do not exist, create them.

    Parameters:
    paths (list of str): List of directory paths.
    """
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
    """
    Perform data augmentation on images and their corresponding masks.

    Parameters:
    stripped_data_path (str): Path to the directory containing the original images.
    augmented_data_path (str): Path to the directory to save the augmented images.
    num_augmentations (int, optional): Number of augmentations to generate per image. Default is 5.
    seed (int, optional): Random seed for reproducibility. Default is 42.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    ensure_dirs([augmented_data_path])

    # List all images
    image_files = [f for f in os.listdir(stripped_data_path) if f.endswith('_image.png')]
    
    # Process each image and its corresponding mask
    for image_file in image_files:
        image_path = os.path.join(stripped_data_path, image_file)
        image = cv2.imread(image_path)
        mask_path = image_path.replace('_image.png', '_mask.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

        # Check if shapes of image and mask match
        if mask is not None and (image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]):
            print(f"Skipping {image_file} due to shape mismatch between image and mask.")
            continue

        for i in range(num_augmentations):
            augmented = augmentation_pipeline(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            aug_image_filename = image_file.replace('_image.png', f'_aug_{i}_image.png')
            aug_mask_filename = image_file.replace('_image.png', f'_aug_{i}_mask.png')

            # Save augmented image and mask
            cv2.imwrite(os.path.join(augmented_data_path, aug_image_filename), aug_image)
            if mask is not None:
                cv2.imwrite(os.path.join(augmented_data_path, aug_mask_filename), aug_mask)

    print(f'Augmentation completed. Augmented data saved to {augmented_data_path}')

# Execute the augmentation
augment_data(stripped_data_path, augmented_data_path)
