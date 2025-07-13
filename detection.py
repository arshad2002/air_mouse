import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, RunningMode
from mouse_controller import MouseController

model_path = os.path.join('.', 'models', 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=RunningMode.VIDEO, 
                                       num_hands=2,
                                       min_hand_detection_confidence=0.5)

landmarker = HandLandmarker.create_from_options(options)


mouse_controller = MouseController()
mouse_control_enabled = True

def set_mouse_control(enabled):
    global mouse_control_enabled
    mouse_control_enabled = enabled


def detected_image(image, frame_timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    
    if not detection_result.hand_landmarks:
        return image
    
    colors = [(255, 0, 0), (0, 0, 255)]
    line_color = (255, 255, 255)
    height, width, _ = image.shape


    for idx, landmarks in enumerate(detection_result.hand_landmarks):
        handedness_label = detection_result.handedness[idx][0].category_name  
        circle_color = colors[0] if handedness_label == 'Left' else colors[1]

    
        if handedness_label == 'Left' and mouse_control_enabled: 
            
            mouse_controller.process_hand(landmarks, width, height)
            
            
            gesture_info = mouse_controller.get_gesture_info(landmarks)
            
            
            status_text = f"Click: {'ON' if gesture_info['is_clicking'] else 'OFF'}"
            pinch_text = f"Pinch: {gesture_info['pinch_distance']:.3f}"
            
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, pinch_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)

       
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x0 = int(landmarks[start_idx].x * width)
            y0 = int(landmarks[start_idx].y * height)
            x1 = int(landmarks[end_idx].x * width)
            y1 = int(landmarks[end_idx].y * height)
            cv2.line(image, (x0, y0), (x1, y1), line_color, 2)

        
        for i, lm in enumerate(landmarks):
            cx = int(lm.x * width)
            cy = int(lm.y * height)
            
            
            if i == 4:  
                cv2.circle(image, (cx, cy), 8, (255, 255, 0), -1)  
            elif i == 8:  
                cv2.circle(image, (cx, cy), 8, (0, 255, 255), -1) 
            elif i == 12:  
                cv2.circle(image, (cx, cy), 8, (255, 0, 255), -1)  
            else:
                cv2.circle(image, (cx, cy), 4, circle_color, -1)

           
            if i == 0:
                control_text = " (CONTROL)" if handedness_label == 'Left' else ""
                cv2.putText(image, handedness_label + control_text, (cx, cy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, circle_color, 2, cv2.LINE_AA)

    
    instructions = [
        "Instructions:",
        "- Use RIGHT hand for control",
        "- Point with INDEX finger to move cursor",
        "- PINCH (thumb+index) to click",
        "- Extend MIDDLE finger and move up/down to scroll",
        "- Press 'q' to quit"
    ]
    
    for i, instruction in enumerate(instructions):
        y_pos = height - (len(instructions) - i) * 25
        cv2.putText(image, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


