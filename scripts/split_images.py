import os
import shutil
import math

def distribute_images(input_folder, n):
    # Get all jpg images in the input folder
    images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    total_images = len(images)
    
    if total_images == 0:
        print("No images found in the input folder.")
        return
    
    # Calculate the number of images per folder
    images_per_folder = math.ceil(total_images / n)
    
    # Create subfolders and distribute images
    base_folder_name = os.path.basename(os.path.normpath(input_folder))
    for i in range(n):
        subfolder_name = f"{base_folder_name}_{i + 1}"
        subfolder_path = os.path.join(input_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Determine the start and end indices for this subfolder
        start_idx = i * images_per_folder
        end_idx = min(start_idx + images_per_folder, total_images)
        
        for j in range(start_idx, end_idx):
            source_path = os.path.join(input_folder, images[j])
            destination_path = os.path.join(subfolder_path, images[j])
            shutil.copy2(source_path, destination_path)
            print(f"Copied {images[j]} to {subfolder_name}")

    print("Image distribution completed.")

# Example usage
input_folder = '/home/ubuntu/shared/clean_frames/Video22'
n = 4  # Number of subfolders to create
distribute_images(input_folder, n)
