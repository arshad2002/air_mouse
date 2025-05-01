from detection import detected_image
import cv2
import time

cap = cv2.VideoCapture(0)
start_time = time.time()
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)
    annotated_frame = detected_image(rgb_frame.copy(), frame_timestamp_ms)
    cv2.imshow("Webcam Video",cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()