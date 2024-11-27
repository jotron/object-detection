from ultralytics import YOLO
import os
import comet_ml
import torch

comet_ml.login()

assert(torch.cuda.is_available())

BATCH_SIZE = -1
IMAGE_SIZE = 1920
EPOCHS = 10

model = YOLO("yolo11m.pt")

# Train the model
train_results = model.train(
    data="datasets/raw/dataset.yaml",  # path to dataset YAML
    epochs=EPOCHS,  # number of training epochs
    imgsz=IMAGE_SIZE,  # training image size
    device=0,  # or cpu
    batch=BATCH_SIZE,
    workers=8,
    plots=True,
    close_mosaic=2,
)