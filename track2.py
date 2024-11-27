from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics.engine.results import Boxes

DETECTION_MODEL = './runs/trainv1/weights/best.pt'
TEST_IMAGES='./test/img1'
OUT_PATH='./runs/custom/video.mp4'

model = YOLO(DETECTION_MODEL)

out = cv2.VideoWriter(
    OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080)
)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict({"track_high_thresh": 0.25,
                  "track_low_thresh": 0.1,
                  "new_track_thresh": 0.6,
                  "track_buffer": 30,
                  "match_thresh": 0.8,
                  "gmc_method": "sparseOptFlow",
                  "proximity_thresh": 0.5,
                  "appearance_thresh": 0.25
                })

tracker = BOTSORT(args=args)
for i, file in enumerate(sorted(os.listdir(TEST_IMAGES))):
    image = os.path.join(TEST_IMAGES, file)
    print(image)
    image = np.array(Image.open(image))
    result = model.predict(image, conf=0.1, imgsz=1920)
    boxes = result[0].boxes
    new_data = boxes.data.clone()

    # Amplify the top ball from 0.1 to 4x as much
    is_ball = boxes.cls == 0
    if (is_ball.any()):
        masked_conf = np.where(is_ball, boxes.conf, -np.inf)
        maximum_ball_index = np.argmax(masked_conf)
        new_data[maximum_ball_index][4] = min(boxes.data[maximum_ball_index][4]*4, 0.95)
        result[0].update(boxes=new_data)

    # Divide confidence in players outside of top 23 by two
    is_player = boxes.cls == 1 
    if is_player.any():
        player_indices = np.where(is_player)[0]
        sorted_indices = player_indices[np.argsort(-boxes.conf[player_indices])]
        # Identify players outside the top 23
        outside_top_23_indices = sorted_indices[23:] if len(sorted_indices) > 23 else []
        # Update confidence for these players
        for idx in outside_top_23_indices:
            new_data[idx][4] *= 0.5
        result[0].update(boxes=new_data)

    image2 = image[:, :, ::-1].copy()
    boxes_tracked = tracker.update(result[0].boxes, image2)

    # Translate boxes back to result
    result[0].boxes = Boxes(boxes_tracked[:, [0, 1, 2, 3, 5, 6]], orig_shape=(1920, 1080))

    # Store to video
    annotated = result[0].plot()
    out.write(annotated)

out.release()
