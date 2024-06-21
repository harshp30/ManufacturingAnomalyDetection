import os
import shutil

# Paths
path = ''
augmented_data_path = f'{path}/augmented_data/classification'
output_data_path = f'{path}/classification_data/classification_images'
labels_file_path = f'{path}/classification_data/labels.txt'

# Ensure the output directory exists
os.makedirs(output_data_path, exist_ok=True)

# Initialize class labels
class_labels = {
    'grid_bent': 0,
    'grid_broken': 1,
    'grid_glue': 2,
    'grid_good': 3,
    'grid_metal_contimation': 4,
    'grid_thread': 5,
    'leather_color': 6,
    'leather_cut': 7,
    'leather_fold': 8,
    'leather_glue': 9,
    'leather_good': 10,
    'leather_poke': 11,
    'metalnut_bent': 12,
    'metalnut_color': 13,
    'metalnut_flip': 14,
    'metalnut_good': 15,
    'metalnut_scratch': 16,
    'screw_good': 17,
    'screw_manipulated_front': 18,
    'screw_scratch_head': 19,
    'screw_scratch_neck': 20,
    'screw_scratch_side': 21,
    'screw_thread_top': 22,
    'cable_bent_wire': 23,
    'cable_cable_swap': 24,
    'cable_combined': 25,
    'cable_cut_inner_insulation': 26,
    'cable_cut_outer_insulation': 27,
    'cable_good': 28,
    'cable_missing_cable': 29,
    'cable_missing_wire': 30,
    'cable_poke_insulation': 31,
    'tile_crack': 32,
    'tile_glue_strip': 33,
    'tile_good': 34,
    'tile_gray_stroke': 35,
    'tile_oil': 36,
    'tile_rough': 37,
    'transistor_bent_lead': 38,
    'transistor_cut_lead': 39,
    'transistor_damaged_case': 40,
    'transistor_good': 41,
    'transistor_misplaced': 42,
    'wood_color': 43,
    'wood_combined': 44,
    'wood_good': 45,
    'wood_hole': 46,
    'wood_liquid': 47,
    'wood_scratch': 48
}

# Process each augmented image and create the labels file
with open(labels_file_path, 'w') as labels_file:
    for filename in os.listdir(augmented_data_path):
        if filename.endswith('.png'):
            # Determine the class by checking if the class name is in the filename
            found_class = None
            for class_name in class_labels:
                if class_name in filename:
                    found_class = class_name
                    break

            if found_class:
                label = class_labels[found_class]
                
                # Copy the image to the output directory
                src_image_path = os.path.join(augmented_data_path, filename)
                dest_image_path = os.path.join(output_data_path, filename)
                shutil.copy(src_image_path, dest_image_path)
                
                # Write the filename and its corresponding label to the labels file
                labels_file.write(f'{filename} {label}\n')
                print(f'Processed file: {filename}, Class: {found_class}, Label: {label}')
            else:
                print(f'No class found for file: {filename}')

print(f'Preprocessing completed. Images saved to {output_data_path} and labels saved to {labels_file_path}')
