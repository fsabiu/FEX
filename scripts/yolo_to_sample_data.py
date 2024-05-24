
from ultralytics import YOLO
import sys
import os

def yolo_to_annotations(imgs_list):

    model = YOLO("../apps/models/best_tanks_militaryTrucks.pt")

    results = model(imgs_list, save_txt=True)

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        
        #print(result.boxes)
    
       


def imgs_list(path):
    img_type = ('.jpg', '.jpeg', '.png')
    
    imgs_list = []
    

    if not os.path.isdir(path):
        print(f"Path {path} is not valid.")
        return imgs_list
    

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(img_type):
                
                full_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(full_path, path)
                custom_path = os.path.join('~', path, relative_path)
                imgs_list.append(custom_path)
    
    return imgs_list


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python train_yolov8.py <DatasetName>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    imgs_list= imgs_list(dataset_path)
    yolo_to_annotations(imgs_list)


    