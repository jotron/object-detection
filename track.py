from ultralytics import YOLO
import os

DETECTION_MODEL = './runs/trainv1/weights/best.pt'

model = YOLO(DETECTION_MODEL)

tracking = model.track("test/4_annotate_1min_bodo_start.mp4", 
                       tracker="track.yaml", 
                       save=True, 
                       show_conf=True,
                       device="cpu", 
                       conf=0.5)