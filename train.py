from ultralytics import YOLO

BATCH_SIZE = 16
IMAGE_SIZE = 640
EPOCHS = 10

model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="datasets/raw/dataset.yaml",  # path to dataset YAML
    epochs=EPOCHS,  # number of training epochs
    imgsz=IMAGE_SIZE,  # training image size
    device="mps",  # or cpu
    batch=BATCH_SIZE,
    workers=8,
    val = False
)