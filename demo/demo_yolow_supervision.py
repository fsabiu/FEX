import os
import cv2
import supervision as sv
from tqdm import tqdm
from inference.models import YOLOWorld

SOURCE_VIDEO_PATH = "media/dron1.mp4"
TARGET_VIDEO_PATH = "media/video-output.mp4"
model = YOLOWorld(model_id="yolo_world/l")
classes = ["car", "train", "building", "bridge", "ship", "pedestrian cross", "pedestrian"]
model.set_classes(classes)
#BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
BOUNDING_BOX_ANNOTATOR = sv.PolygonAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=0.5, text_scale=0.3, text_color=sv.Color.BLACK)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(generator)
sv.plot_image(frame, (10, 10))
results = model.infer(frame, confidence=0.002)
detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
annotated_image = frame.copy()
annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
sv.plot_image(annotated_image, (10, 10))
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(video_info)
width, height = video_info.resolution_wh
frame_area = width * height
print(frame_area)
results = model.infer(frame, confidence=0.002)
detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
print(detections.area)
detections = detections[(detections.area / frame_area) < 0.10]
annotated_image = frame.copy()
annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
sv.plot_image(annotated_image, (10, 10))
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
width, height = video_info.resolution_wh
frame_area = width * height
print(frame_area)
with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
    results = None
    for i, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
        if i%3 == 0:
            results = model.infer(frame, confidence=0.002)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
        detections = detections[(detections.area / frame_area) < 0.10]
        annotated_frame = frame.copy()
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        sink.write_frame(annotated_frame)









