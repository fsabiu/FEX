from ultralytics import YOLO
import cv2
import datetime
from datetime import datetime
import numpy as np
import supervision as sv
import time
#from google.colab.patches import cv2_imshow

#initialize a YOLO-World model
model = YOLO('yolov8l-world.pt')    # or choose yolov8m/l-world.pt
#custom classes
model.set_classes(["car", "pedestrian crossing", "train", "ship"])
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
cap = cv2.VideoCapture("media/dron1.mp4")
w,h,fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w, h, fps)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w,h))
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    print("---- img type")
    print(type(img))
    print(np.shape(img))
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    results = model.predict(img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Real inference time: " + str(elapsed_time) + " seconds")
    #results = model.infer(img, confidence=0.002)
    detections = sv.Detections.from_ultralytics(results[0])
    annotated_frame = bounding_box_annotator.annotate(
        scene = img.copy(),
        detections = detections[detections.confidence > 0.001]
    )
    #annotated_frame = label_annotator.annotate(
    #    scene=annotated_frame, detections = detections
    #)
    polygon_annotator = sv.PolygonAnnotator()
    annotated_frame = polygon_annotator.annotate(
    scene=annotated_frame,
    detections=detections
    )
    out.write(annotated_frame)
    #cv2.imshow("Image", annotated_frame)
    #cv2_imshow(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
