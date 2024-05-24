import cv2
import os

def flip_image_and_update_yolo(image_path, yolo_annotation_path, output_image_path, output_annotation_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    
    # Load YOLO annotations
    with open(yolo_annotation_path, 'r') as file:
        annotations = file.readlines()
    
    updated_annotations = []
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Update x_center for the flipped image
        new_x_center = 1.0 - x_center
        
        # Create the updated annotation line
        updated_annotation = f"{class_id} {new_x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        updated_annotations.append(updated_annotation)
    
    # Save the flipped image
    cv2.imwrite(output_image_path, flipped_image)

    # Save the updated annotations
    with open(output_annotation_path, 'w') as file:
        for annotation in updated_annotations:
            file.write(f"{annotation}\n")

def process_directory(input_image_dir, input_annotation_dir, output_image_dir, output_annotation_dir):
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)
    
    # Loop over all images in the input image directory
    for image_filename in os.listdir(input_image_dir):
        if image_filename.endswith(('.jpg', '.jpeg', '.png')):  # Ensure correct image file types
            image_path = os.path.join(input_image_dir, image_filename)
            annotation_filename = os.path.splitext(image_filename)[0] + '.txt'
            annotation_path = os.path.join(input_annotation_dir, annotation_filename)

            # Define output filenames with "_flipped"
            output_image_filename = os.path.splitext(image_filename)[0] + '_flipped' + os.path.splitext(image_filename)[1]
            output_annotation_filename = os.path.splitext(annotation_filename)[0] + '_flipped.txt'

            # Define output paths
            output_image_path = os.path.join(output_image_dir, output_image_filename)
            output_annotation_path = os.path.join(output_annotation_dir, output_annotation_filename)
            
            # Process the image and its annotations
            flip_image_and_update_yolo(image_path, annotation_path, output_image_path, output_annotation_path)

# Define input and output directories
input_image_dir = '/home/ubuntu/shared/data/Day2_sample_data/train/images'
input_annotation_dir = '/home/ubuntu/shared/data/Day2_sample_data/train/labels'
output_image_dir = input_image_dir
output_annotation_dir = input_annotation_dir

# Process the entire directory
process_directory(input_image_dir, input_annotation_dir, output_image_dir, output_annotation_dir)
