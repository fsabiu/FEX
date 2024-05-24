from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as numpy

class Detector: 
    def __init__(self, model_type="PS"):  # Por defecto, usaré el modelo de segmentación panóptica
        self.cfg = get_cfg()
        self.model_type = model_type

        # Configuración del modelo según el tipo
        if model_type == "PS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        # Establecer umbral de confianza
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" 

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

        # Visualizar las máscaras sin los nombres de las clases
        viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
        viz._default_font_size = 15
        out = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        cv2.imwrite("images/output.jpg", out.get_image()[:, :, ::-1])

    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening the file")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_filename = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        success, image = cap.read()

        while success:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]

            # Visualizar las máscaras sin los nombres de las clases
            viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2, instance_mode=ColorMode.SEGMENTATION)
            viz._default_font_size = 15
            out_frame = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo).get_image()[:, :, ::-1]

            out.write(out_frame)
            success, image = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
