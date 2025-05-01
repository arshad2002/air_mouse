import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, RunningMode

model_path = os.path.join('.', 'models', 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=RunningMode.VIDEO, 
                                       num_hands=2,
                                       min_hand_detection_confidence=0.5)

landmarker = HandLandmarker.create_from_options(options)


def draw_landmarks_on_image(image, frame_timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    
    if not detection_result.hand_landmarks:
        return image
    colors = [(0, 255, 0), (0, 0, 255)]

    for idx, landmarks in enumerate(detection_result.hand_landmarks):
        color = colors[idx % len(colors)]

        height, width, _ = image.shape
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x0 = int(landmarks[start_idx].x * width)
            y0 = int(landmarks[start_idx].y * height)
            x1 = int(landmarks[end_idx].x * width)
            y1 = int(landmarks[end_idx].y * height)
            cv2.line(image, (x0, y0), (x1, y1), color, 2)

        for lm in landmarks:
            cx = int(lm.x * width)
            cy = int(lm.y * height)
            cv2.circle(image, (cx, cy), 4, color, -1)

    return image


