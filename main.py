from detection import detected_image, set_mouse_control
import cv2
import time
import pyautogui

# Disable pyautogui failsafe initially (can be re-enabled)
pyautogui.FAILSAFE = True

print("=== Air Mouse Controller ===")
print("Instructions:")
print("- Use your RIGHT hand for mouse control")
print("- Point with INDEX finger to move cursor")
print("- PINCH (thumb + index finger) to click")
print("- Extend MIDDLE finger and move up/down to scroll")
print("- Press 'q' to quit")
print("- Press 's' to toggle mouse control on/off")
print("- Press 'f' to toggle failsafe on/off")
print("\nStarting camera...")

cap = cv2.VideoCapture(0)
start_time = time.time()
mouse_control_enabled = True
show_instructions = True

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_timestamp_ms = int((time.time() - start_time) * 1000)
    
    # Process frame for hand detection and mouse control
    annotated_frame = detected_image(rgb_frame.copy(), frame_timestamp_ms)
    
    # Add status indicators
    status_color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
    status_text = "MOUSE: ON" if mouse_control_enabled else "MOUSE: OFF"
    cv2.putText(annotated_frame, status_text, (10, annotated_frame.shape[0] - 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    
    failsafe_text = f"FAILSAFE: {'ON' if pyautogui.FAILSAFE else 'OFF'}"
    cv2.putText(annotated_frame, failsafe_text, (10, annotated_frame.shape[0] - 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Air Mouse Controller", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        mouse_control_enabled = not mouse_control_enabled
        set_mouse_control(mouse_control_enabled)
        if mouse_control_enabled:
            print("Mouse control ENABLED")
        else:
            print("Mouse control DISABLED")
    elif key == ord('f'):
        pyautogui.FAILSAFE = not pyautogui.FAILSAFE
        if pyautogui.FAILSAFE:
            print("Failsafe ENABLED (move mouse to corner to stop)")
        else:
            print("Failsafe DISABLED")
    elif key == ord('h'):
        show_instructions = not show_instructions
        print(f"Instructions {'shown' if show_instructions else 'hidden'}")

print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Air Mouse Controller stopped.")