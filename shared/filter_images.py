import os
import shutil
import sys
from PIL import Image


def object_too_large(annotation, image_width, image_height):
    # YOLOv8 format: class x_center y_center width height
    _, _, _, width, height = map(float, annotation.split())
    obj_width = width * image_width
    obj_height = height * image_height
    return obj_width > 0.5 * image_width or obj_height > 0.5 * image_height

def process_images(images_folder, annotations_folder):


    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith('.txt'):
            annotation_path = os.path.join(annotations_folder, annotation_file)
            with open(annotation_path, 'r') as file:
                annotations = file.readlines()
            

            image_file = annotation_file.replace('.txt', '.jpg') 
            image_path = os.path.join(images_folder, image_file)
            
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image_width, image_height = image.size


                remove_image = False
                for annotation in annotations:
                    if object_too_large(annotation, image_width, image_height):
                        remove_image = True
                        break
                
                if remove_image:
                    os.remove(image_path)
                    os.remove(annotation_path)
                    print(f'Removed {image_file} and its annotation')

    print('Processing complete.')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Use: python filter_images.py path/to/images/ path/to/annotations")
        sys.exit(1)


    annotations_folder = sys.argv[2] 
    images_folder = sys.argv[1]

    process_images(images_folder, annotations_folder)