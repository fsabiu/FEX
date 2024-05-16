import sys
import os
import json
import shutil
import glob

def process_dataset(dataset_names, dataset_tgt_folder, datasets_dict):
    
    
    for dataset_name in dataset_names:
        dataset_info = datasets_dict.get(dataset_name)
        if dataset_info:
            create_folder_structure( dataset_tgt_folder)
            copy_data(dataset_info, dataset_tgt_folder)
        else:
            print(f"Dataset '{dataset_name}' not found in datasets_dict.json")

def create_folder_structure( dataset_tgt_folder):
    base_path = os.path.expanduser("/shared/")
    #dataset_path = os.path.join(base_path, "datasets_for_training/"+str(dataset_tgt_folder)+"/"+str(dataset_name))
    dataset_path = os.path.join(base_path, "datasets_for_training/"+str(dataset_tgt_folder))
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    folders = [train_path, val_path, test_path]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

def copy_data(dataset_info, dataset_tgt_folder):
    base_path = os.path.expanduser("/shared/")
    src_dir = os.path.join(base_path, "data", dataset_info["dir_name"])
    #tgt_dir = os.path.join(base_path, "datasets_for_training/", str(dataset_tgt_folder), dataset_info["dir_name"])
    
    for phase in ["train", "val", "test"]:
        src_img_dir = os.path.join(src_dir, dataset_info[phase + "_name"], dataset_info[phase + "_img"])
        src_label_dir = os.path.join(src_dir, dataset_info[phase + "_name"], dataset_info[phase + "_labels"])

        tgt_img_dir = os.path.join(base_path, "datasets_for_training",  str(dataset_tgt_folder), phase, "images")
        tgt_label_dir = os.path.join(base_path, "datasets_for_training",  str(dataset_tgt_folder), phase, "labels")

        print(src_img_dir, tgt_img_dir)
        copy_files(src_img_dir, tgt_img_dir, dataset_info["dir_name"])
        
        copy_files(src_label_dir, tgt_label_dir, dataset_info["dir_name"])       
        print(src_label_dir, tgt_label_dir)  

def copy_files(src_dir, tgt_dir, dataset_dir_name):
    if os.path.exists(src_dir):
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)

        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            tgt_file = os.path.join(tgt_dir, f"{dataset_dir_name}_{filename}")
            shutil.copy(src_file, tgt_file)        

def change_labels_classes(datasets_dict):
    changes = {}
    for dataset_name, dataset_info in datasets_dict.items():
        id_classes = dataset_info.get("id_classes", {})
        dir_name = dataset_info.get("dir_name")

        if dir_name not in changes:
            changes[dir_name] = {}

        for class_name, class_info in id_classes.items():
            for class_id, new_id in class_info.items():
                if int(class_id) != new_id:
                    print(f"Dataset '{dataset_name}', dir_name '{dir_name}': Class '{class_name}' changed from id '{class_id}' to '{new_id}'")
                    if class_name not in changes[dir_name]:
                        changes[dir_name][class_name] = {}
                    changes[dir_name][class_name][class_id] = new_id

    return changes

def modify_labels(dataset_tgt_folder, changes):
    base_path = os.path.expanduser("/shared/")

    dataset_path = os.path.join(base_path, "datasets_for_training", "YOLO_" + str(dataset_tgt_folder))

    transformed_changes = {}
    for class_changes in changes.values():
        for new_id in class_changes.values():
            transformed_changes.update(new_id)

    #print(transformed_changes)

    for dir_name, class_changes in changes.items():
        for phase in ["train", "val", "test"]:
            label_dir = os.path.join(dataset_path, phase, "labels")

            for filename in os.listdir(label_dir):
                if filename.startswith(dir_name):
                    file_path = os.path.join(label_dir, filename)
                    if bool(class_changes):
                        #print(file_path, transformed_changes)
                        update_labels_in_file(file_path, transformed_changes)

                
                

def update_labels_in_file(file_path, class_changes):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            parts = line.split()
            class_id = parts[0]
            if class_id in class_changes:
                new_id = class_changes[class_id]
                if new_id != -1:
                    parts[0] = str(new_id)
                    f.write(' '.join(parts) + '\n')
            else:
                f.write(line) 

def remove_empty_files_recursive(dataset_tgt_folder):
    images_dir = os.path.join("/shared/datasets_for_training/YOLO_" + dataset_tgt_folder)
    print("Deleting images without objects...")
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            # verify if it is a .txt
            
            if file.endswith(".txt"):
                txt_file = os.path.join(root, file)
                # verify if it is empty
                with open(txt_file, 'r') as f:
                    if not f.read().strip():
                        # drop .txt
                        os.remove(txt_file)
                        images_dir = os.path.join(os.path.dirname(root), "images")
                        # drop related image
                        image_pattern = os.path.splitext(file)[0] + ".*"  
                        image_files = glob.glob(os.path.join(images_dir, image_pattern))
                        valid_extensions = [".jpg", ".jpeg", ".png"]
                        for image_file in image_files:
                            if os.path.splitext(image_file)[1].lower() in valid_extensions:
                                os.remove(image_file)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python script.py <DatasetName>")
        sys.exit(1)
    with open('datasets_dict.json', 'r') as f:
        datasets_dict = json.load(f)
    dataset_names = sys.argv[1:-1]
    dataset_tgt_folder = sys.argv[-1]

    process_dataset(dataset_names, "YOLO_"+str(dataset_tgt_folder), datasets_dict)
    changes = change_labels_classes(datasets_dict)
    modify_labels(dataset_tgt_folder, changes)
    remove_empty_files_recursive(dataset_tgt_folder)