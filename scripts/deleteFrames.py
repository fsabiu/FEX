import os
import shutil

def keep_every_third_file(directory):
    # Create a directory to store every 3rd file
    output_dir = os.path.join(directory, 'every_3rd')
    os.makedirs(output_dir, exist_ok=True)

    # List all files in the directory
    files = [f for f in sorted(os.listdir(directory)) if os.path.isfile(os.path.join(directory, f))]

    # Iterate over the files and move every 3rd file
    for i, file in enumerate(files):
        file_path = os.path.join(directory, file)
        if (i + 1) % 3 == 0:
            # Move the file to the every_3rd directory
            shutil.move(file_path, os.path.join(output_dir, file))
        else:
            # Delete the file
            os.remove(file_path)

# Specify the directory you want to process
directory = "/home/ubuntu/shared/clean_frames/2105"  # Current directory
keep_every_third_file(directory)
