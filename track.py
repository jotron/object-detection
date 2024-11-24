from ultralytics import YOLO

DETECTION_MODEL = './runs/train/best.pt'

model = YOLO(DETECTION_MODEL)

tracking = model.track("RBK_TDT17/4_annotate_1min_bodo_start/4_annotate_1min_bodo_start.mp4", save=True, show_conf=True, device='mps', conf=0.1)