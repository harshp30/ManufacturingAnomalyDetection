import os
import shutil
from PIL import Image

# Paths
base_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/data/mvtec_anomaly_detection/cable'
stripped_data_path = '/home/paperspace/Projects/ManufacturingAnomalyDetection/anomaly_data/classification'

# Ensure directories exist
def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Function to create a blank mask of specific size
def create_blank_mask(size, output_path):
    blank_mask = Image.new('L', size, 0)
    blank_mask.save(output_path)

# Function to organize and rename files, and create masks where needed
def organize_and_rename_files(base_path, stripped_data_path):
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

    # Processing train, test, and ground_truth directories
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

                        shutil.copy(src_path, dest_image_path)
                        counters[category] += 1

                        if category == 'good':
                            # Get the size of the image
                            with Image.open(src_path) as img:
                                image_size = img.size
                            
                            # Create a blank mask with the same size as the image
                            dest_mask_path = os.path.join(stripped_data_path, f'{new_file_name}_mask.png')
                            create_blank_mask(image_size, dest_mask_path)
                        else:
                            # Assuming mask files are in a 'ground_truth' subdirectory
                            ground_truth_path = os.path.join(base_path, 'ground_truth', category, file_name.replace('.png', '_mask.png'))
                            if os.path.exists(ground_truth_path):
                                dest_mask_path = os.path.join(stripped_data_path, f'{new_file_name}_mask.png')
                                shutil.copy(ground_truth_path, dest_mask_path)
                            else:
                                print(f"Warning: No mask found for {file_name} in {category}. Creating a blank mask.")
                                dest_mask_path = os.path.join(stripped_data_path, f'{new_file_name}_mask.png')
                                create_blank_mask(image_size, dest_mask_path)

    print(f'Files organized and renamed in {stripped_data_path}')

# Execute the file organization and renaming
organize_and_rename_files(base_path, stripped_data_path)
