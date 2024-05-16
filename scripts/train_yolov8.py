import json
import os
import yaml
import sys
from ultralytics import YOLO

def json_to_yolo_yaml(dataset_path):
    # Read the JSON file
    with open(dataset_path + 'classes.json', 'r') as file:
        data = json.load(file)

    # Prepare the content for the YAML file
    yolo_yaml_content = {
        'path': dataset_path,
        'train': os.path.join(dataset_path, 'train/images'),
        'val': os.path.join(dataset_path, 'val/images'),
        'names': {int(k): v for k, v in data.items()}
    }
    
    # Write the YAML file
    yaml_path = os.path.join(dataset_path, 'train.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(yolo_yaml_content, file, default_flow_style=False)
    
    return yaml_path

def train_yolo():
    model = YOLO('yolov8x.pt')
    print(model.info())
    results = model.train(data='train.yaml', epochs=100, imgsz=640)
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python train_yolov8.py <DatasetName>")
        sys.exit(1)

    dataset = sys.argv[1]
    something_list = sys.argv[1:-1]
    last_argument = sys.argv[-1]
    dataset_path = '/shared/datasets_for_training/' + dataset + '/'

    yaml_path = json_to_yolo_yaml(dataset_path)

    train_yolo()

