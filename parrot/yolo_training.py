from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=1024, device=[0, 1])