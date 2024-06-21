import os
import shutil
from PIL import Image

# Paths
path = ''
base_path = f'{path}/data/mvtec_anomaly_detection/wood'
stripped_data_path = f'{path}/anomaly_data/classification'

def ensure_dirs(paths):
    """
    Ensure that the directories in the given list exist.
    If a directory does not exist, create it.

    Parameters:
    paths (list of str): List of directory paths to check and create if needed.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def create_blank_mask(size, output_path):
    """
    Create a blank mask of the specified size and save it to the output path.

    Parameters:
    size (tuple): Size of the mask (width, height).
    output_path (str): Path where the blank mask will be saved.
    """
    blank_mask = Image.new('L', size, 0)  # Create a blank (black) mask
    blank_mask.save(output_path)  # Save the blank mask

def organize_and_rename_files(base_path, stripped_data_path):
    """
    Organize and rename image files from the base path and save them to the stripped data path.
    Also create blank masks where needed.

    Parameters:
    base_path (str): Path to the base directory containing the original images.
    stripped_data_path (str): Path to the directory where organized images will be saved.
    """
    ensure_dirs([stripped_data_path])  # Ensure the target directory exists

    # Initialize counters for different types of flaws including 'good'
    counters = {
        'good': 0,
        'color': 0,
        'combined': 0,
        'hole': 0,
        'liquid': 0,
        'scratch': 0
    }

    # Process train and test directories
    for sub_dir in ['train', 'test']:
        current_dir = os.path.join(base_path, sub_dir)

        for category in os.listdir(current_dir):
            category_path = os.path.join(current_dir, category)

            if os.path.isdir(category_path):  # Check if it is a directory
                for file_name in os.listdir(category_path):
                    if file_name.endswith('.png'):  # Check if it is a PNG image
                        src_path = os.path.join(category_path, file_name)
                        new_file_name = f'{category}_{counters[category]:03d}'  # Create a new file name
                        dest_image_path = os.path.join(stripped_data_path, f'{new_file_name}_image.png')

                        shutil.copy(src_path, dest_image_path)  # Copy the image to the new location
                        counters[category] += 1  # Increment the counter for the category

    print(f'Files organized and renamed in {stripped_data_path}')  # Print completion message

# Execute the file organization and renaming
organize_and_rename_files(base_path, stripped_data_path)  # Call the function with specified paths
