import os
import sys
import shutil
import random

def split_dataset(dataset_name, image_folder, labels_folder):
    
    val_image_folder = "/shared/data/"+str(dataset_name) +'/val/images'
    test_image_folder ="/shared/data/"+str(dataset_name) + '/test/images'
    val_label_folder = "/shared/data/"+str(dataset_name) +'/val/labels'
    test_label_folder = "/shared/data/"+str(dataset_name) +'/test/labels'
    
    # Create folders
    os.makedirs( str(val_image_folder), exist_ok=True)
    os.makedirs(str(test_image_folder), exist_ok=True)
    os.makedirs(str(val_label_folder), exist_ok=True)
    os.makedirs(str(test_label_folder), exist_ok=True)
    
    
    #print(image_folder)
    image_files = os.listdir(image_folder)
    
    
    num_images = len(image_files)
    num_val_images = int(0.25 * num_images)
    num_test_images = int(0.05 * num_images)
    
    # Select random images
    val_images = random.sample(image_files, num_val_images)
    remaining_images = [img for img in image_files if img not in val_images]
    test_images = random.sample(remaining_images, num_test_images)
    
    # Copy images/labels
    
    for image in val_images:
        shutil.move(os.path.join(image_folder, image), val_image_folder)
        shutil.move(os.path.join(labels_folder, image.replace('.jpg', '.txt')), val_label_folder)
    
    for image in test_images:
        shutil.move(os.path.join(image_folder, image), test_image_folder)
        shutil.move(os.path.join(labels_folder, image.replace('.jpg', '.txt')), test_label_folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python train_yolov8.py <DatasetName> <DatasetName/image_folder_name> <DatasetName/labels_folder_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    images_folder = sys.argv[2]
    labels_folder = sys.argv[3]
    
    images_folder_path = '/shared/data/' + images_folder
    labels_folder_path = '/shared/data/' + labels_folder

split_dataset(dataset_name, images_folder_path, labels_folder_path)