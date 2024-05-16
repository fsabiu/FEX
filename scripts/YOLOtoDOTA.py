# YOLO folders structure train, val, test with images, labels
import json
from PIL import Image
import shutil
import os
import sys

def check_dataset(dataset_path):

    # Check if the name after the last slash starts with YOLO_
    name_after_last_slash = dataset_path.split('/')[-1]
    if not name_after_last_slash.startswith('YOLO_'):
        print("Error: Dataset name should start with 'YOLO_'")
        return False

    # Check if necessary folders exist
    required_folders = [
        'train/images',
        'train/labels',
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]

    # Checking folders
    for folder in required_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder}' does not exist.")
            return False

    # Checking class dictionary
    # Check if classes.json exists
    classes_json_path = os.path.join(dataset_path, 'classes.json')
    if not os.path.exists(classes_json_path):
        print("Error: 'classes.json' file does not exist.")
        return False, {}

    # Load the content of classes.json into a dictionary
    try:
        with open(classes_json_path, 'r') as f:
            classes_dict = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode 'classes.json': {e}")
        return False, {}

    return True, classes_dict

def create_sibling_folder(dataset_path):
    # Extract YOLO dataset name
    yolo_dataset_name = dataset_path.split('/')[-1]
    
    # Extract dataset name from YOLO dataset name
    dataset_name = yolo_dataset_name.split('YOLO_')[-1]

    # Create sibling folder with name "DOTA_" + dataset_name
    parent_folder = os.path.dirname(dataset_path)
    sibling_folder_name = "DOTA_" + dataset_name
    sibling_folder_path = os.path.join(parent_folder, sibling_folder_name)

    try:
        os.makedirs(sibling_folder_path)
        print(f"Sibling folder '{sibling_folder_name}' created successfully.")
    except FileExistsError:
        print(f"Error: Sibling folder '{sibling_folder_name}' already exists.")

    # Create train, val, and test folders with images and labels
    for folder_name in ['train', 'val', 'test']:
        folder_path = os.path.join(sibling_folder_path, folder_name)
        print("creating " + os.path.join(folder_path, 'images'))
        os.makedirs(os.path.join(folder_path, 'images'))
        os.makedirs(os.path.join(folder_path, 'labels'))
        print(f"Folder '{folder_name}' created successfully with images and labels.")

    return sibling_folder_path

def change_extension_to_png(image_path):
    # Split the path into root and extension
    root, ext = os.path.splitext(image_path)
    # Append '.png' to the root to form the new path
    new_path = root + '.png'
    return new_path

def copy_images_and_generate_labels(dataset_path, sibling_folder_path, class_dict):
    # Define folders to copy images from and to
    folders_to_copy = ['train', 'val', 'test']

    # Iterate over folders and copy images, generate labels
    for folder in folders_to_copy:
        label_source_folder = os.path.join(dataset_path, folder, 'labels')
        image_source_folder = os.path.join(dataset_path, folder, 'images')
        label_destination_folder = os.path.join(sibling_folder_path, folder, 'labels')
        image_destination_folder = os.path.join(sibling_folder_path, folder, 'images')

        for i, image_filename in enumerate(os.listdir(image_source_folder)):
            # Open the image and get its size
            with Image.open(os.path.join(image_source_folder, image_filename)) as img:
                img_width, img_height = img.size
            
                # Copy the image to the destination folder
                #source_image_path = os.path.join(image_source_folder, image_filename)
                destination_image_path = os.path.join(image_destination_folder, image_filename)
                destination_image_png_path = change_extension_to_png(destination_image_path)
                img.save(destination_image_png_path, 'PNG')
            #shutil.copy(source_image_path, destination_image_path)
            if i % 50 == 0:
                print(f"Image file nr. '{i+1}' copied from '{image_source_folder}' to '{image_destination_folder}'.")

            # Look for the label with the same name (except the extension) in the source folder
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            source_label_path = os.path.join(label_source_folder, label_filename)
            if os.path.exists(source_label_path):
                destination_label_path = os.path.join(label_destination_folder, label_filename)
                write_modified_label(source_label_path, destination_label_path, img_width, img_height, class_dict)
                if i % 50 == 0:
                    print(f"Label file nr. '{i+1}' converted for '{label_destination_folder}'.")

def write_modified_label(source_label_path, destination_label_path, img_width, img_height, class_dict):
    # Read the original label content
    with open(source_label_path, 'r') as f:
        original_label_content = f.read()

    # Calculate relative coordinates based on original image size and write to new label file
    with open(destination_label_path, 'w') as f:
        for line in original_label_content.splitlines():
            if line:
                values = line.split()
                class_id = class_dict[values[0]]
                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])
                
                # Convert relative coordinates to absolute coordinates
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center - height / 2) * img_height)
                x3 = int((x_center + width / 2) * img_width)
                y3 = int((y_center + height / 2) * img_height)
                x4 = int((x_center - width / 2) * img_width)
                y4 = int((y_center + height / 2) * img_height)

                # Write DOTA format to new label file
                dota_line = f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {class_id} 0\n"
                f.write(dota_line)

def remove_points(folder_path):
    # Recursively traverse the directory
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Find the position of the last dot
            last_dot_position = filename.rfind('.')
            
            if last_dot_position != -1:
                # Split the filename into name and extension parts
                name = filename[:last_dot_position]
                extension = filename[last_dot_position:]
                
                # Remove all dots from the name part
                new_name = name.replace('.', '') + extension
                
                # Construct full old and new paths
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_name)
                
                # Rename the file
                os.rename(old_path, new_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python YOLOtoDOTA.py dataset_path")
        sys.exit(1)

    dataset_path = sys.argv[1]
    # Remove trailing slash if present
    if dataset_path.endswith('/'):
        dataset_path = dataset_path[:-1]

    # Checking folders structure
    check, class_dict = check_dataset(dataset_path)
    if check:
        print("Dataset validation successful.")
        # Creating DOTA_folder
        sibling_folder_path = create_sibling_folder(dataset_path)
        copy_images_and_generate_labels(dataset_path, sibling_folder_path, class_dict)
        remove_points(sibling_folder_path)