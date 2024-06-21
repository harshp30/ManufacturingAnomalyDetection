import os
import shutil
from PIL import Image

# Paths
path = ''
base_path = f'{path}/data/mvtec_anomaly_detection/cable'
stripped_data_path = f'{path}/anomaly_data/classification'

# Ensure directories exist
def ensure_dirs(paths):
    """
    Ensure that the directories in the given list exist. If they do not exist, create them.

    Parameters:
    paths (list of str): List of directory paths.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to create a blank mask of specific size
def create_blank_mask(size, output_path):
    """
    Create a blank mask image of a specific size and save it to the given path.

    Parameters:
    size (tuple): Size of the mask (width, height).
    output_path (str): Path to save the blank mask image.
    """
    blank_mask = Image.new('L', size, 0)
    blank_mask.save(output_path)

# Function to organize and rename files, and create masks where needed
def organize_and_rename_files(base_path, stripped_data_path):
    """
    Organize and rename image files, and create corresponding masks where needed.

    Parameters:
    base_path (str): Base directory containing the source image files.
    stripped_data_path (str): Target directory to save the organized image files and masks.
    """
    ensure_dirs([stripped_data_path])  # Ensure the target directory exists

    # Initialize counters for different types of flaws including 'good'
    counters = {
        'good': 0,
        'bent_wire': 0,
        'cable_swap': 0,
        'combined': 0,
        'cut_inner_insulation': 0,
        'cut_outer_insulation': 0,
        'missing_cable': 0,
        'missing_wire': 0,
        'poke_insulation': 0
    }

    # Process train and test directories
    for sub_dir in ['train', 'test']:
        current_dir = os.path.join(base_path, sub_dir)

        for category in os.listdir(current_dir):
            category_path = os.path.join(current_dir, category)

            if os.path.isdir(category_path):
                for file_name in os.listdir(category_path):
                    if file_name.endswith('.png'):
                        src_path = os.path.join(category_path, file_name)
                        new_file_name = f'{category}_{counters[category]:03d}'
                        dest_image_path = os.path.join(stripped_data_path, f'{new_file_name}_image.png')

                        # Copy the image file to the new location with a new name
                        shutil.copy(src_path, dest_image_path)
                        counters[category] += 1

                        if category == 'good':
                            # For 'good' category, create a blank mask with the same size as the image
                            with Image.open(src_path) as img:
                                image_size = img.size
                            dest_mask_path = os.path.join(stripped_data_path, f'{new_file_name}_mask.png')
                            create_blank_mask(image_size, dest_mask_path)
                        else:
                            # For other categories, copy the corresponding ground truth mask if it exists
                            ground_truth_path = os.path.join(base_path, 'ground_truth', category, file_name.replace('.png', '_mask.png'))
                            if os.path.exists(ground_truth_path):
                                dest_mask_path = os.path.join(stripped_data_path, f'{new_file_name}_mask.png')
                                shutil.copy(ground_truth_path, dest_mask_path)
                            else:
                                # If no ground truth mask is found, create a blank mask
                                print(f"Warning: No mask found for {file_name} in {category}. Creating a blank mask.")
                                dest_mask_path = os.path.join(stripped_data_path, f'{new_file_name}_mask.png')
                                create_blank_mask(image_size, dest_mask_path)

    print(f'Files organized and renamed in {stripped_data_path}')

# Execute the file organization and renaming
organize_and_rename_files(base_path, stripped_data_path)
