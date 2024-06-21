import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
path = ''
input_data_path = f'{path}/classification_data/classification_images'
labels_file_path = f'{path}/classification_data/labels.txt'
output_data_path = f'{path}/training_data/classification'

# Ensure directories exist
def ensure_dirs(paths):
    """
    Ensure that the specified directories exist. If they do not, create them.

    Parameters:
    paths (list): List of directory paths to check and create if necessary.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to split data into training, validation, and testing sets
def split_data(input_data_path, labels_file_path, output_data_path):
    """
    Split data into training, validation, and testing sets, and save them to the specified output directory.

    Parameters:
    input_data_path (str): Path to the directory containing input images.
    labels_file_path (str): Path to the file containing image labels.
    output_data_path (str): Path to the directory to save the split data.
    """
    # Ensure training, validation, and testing directories exist
    ensure_dirs([
        os.path.join(output_data_path, 'train', 'images'),
        os.path.join(output_data_path, 'val', 'images'),
        os.path.join(output_data_path, 'test', 'images')
    ])
    
    # Read labels from file
    images = []
    labels = []
    with open(labels_file_path, 'r') as f:
        for line in f:
            filename, label = line.strip().split()
            images.append(filename)
            labels.append(int(label))
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Split into training, validation, and testing sets (60% train, 20% val, 20% test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.4, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42)
    
    # Save images and labels
    def save_images(images, labels, set_name):
        """
        Save images and their corresponding labels to the specified set (train, val, or test).

        Parameters:
        images (array): Array of image filenames.
        labels (array): Array of corresponding labels.
        set_name (str): Name of the set (train, val, or test) to save the images and labels to.
        """
        set_dir = os.path.join(output_data_path, set_name, 'images')
        labels_path = os.path.join(output_data_path, set_name, 'labels.txt')
        with open(labels_path, 'w') as f:
            for i in range(len(images)):
                src_image_path = os.path.join(input_data_path, images[i])
                dest_image_path = os.path.join(set_dir, images[i])
                shutil.copy(src_image_path, dest_image_path)
                f.write(f'{images[i]} {labels[i]}\n')

    # Save train, validation, and test sets
    save_images(train_images, train_labels, 'train')
    save_images(val_images, val_labels, 'val')
    save_images(test_images, test_labels, 'test')

    print(f'Data split into training, validation, and testing sets and saved to {output_data_path}.')

# Execute the data splitting
split_data(input_data_path, labels_file_path, output_data_path)
