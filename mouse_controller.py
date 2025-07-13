import pyautogui
import numpy as np
import time
import math
from collections import deque

class MouseController:
    def __init__(self, screen_width=None, screen_height=None, smoothing_factor=0.7):
        """
        Initialize the mouse controller
        
        Args:
            screen_width: Screen width (auto-detected if None)
            screen_height: Screen height (auto-detected if None)
            smoothing_factor: Smoothing factor for cursor movement (0-1)
        """
        # Get screen dimensions
        if screen_width is None or screen_height is None:
            self.screen_width, self.screen_height = pyautogui.size()
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
        
        # Mouse control settings
        self.smoothing_factor = smoothing_factor
        self.click_threshold = 0.05  # Distance threshold for click detection
        self.scroll_threshold = 0.1  # Distance threshold for scroll detection
        
        # State tracking
        self.prev_cursor_pos = None
        self.cursor_history = deque(maxlen=5)  # For smoothing
        self.is_clicking = False
        self.click_start_time = 0
        self.click_duration_threshold = 0.3  # 300ms minimum click duration
        self.last_click_time = 0
        self.click_cooldown = 0.5  # 500ms between clicks
        
        # Gesture state
        self.prev_pinch_distance = None
        self.scroll_start_time = 0
        
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01  
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_pinch_distance(self, thumb_tip, index_tip):
        return self.calculate_distance(
            (thumb_tip.x, thumb_tip.y),
            (index_tip.x, index_tip.y)
        )
    
    def smooth_position(self, new_pos):
        if self.prev_cursor_pos is None:
            self.prev_cursor_pos = new_pos
            return new_pos
        
        # Exponential smoothing
        smoothed_x = self.smoothing_factor * self.prev_cursor_pos[0] + (1 - self.smoothing_factor) * new_pos[0]
        smoothed_y = self.smoothing_factor * self.prev_cursor_pos[1] + (1 - self.smoothing_factor) * new_pos[1]
        
        smoothed_pos = (smoothed_x, smoothed_y)
        self.prev_cursor_pos = smoothed_pos
        return smoothed_pos
    
    def convert_to_screen_coords(self, hand_x, hand_y, frame_width, frame_height):
        """
        Convert hand coordinates to screen coordinates
        
        Args:
            hand_x, hand_y: Normalized hand coordinates (0-1)
            frame_width, frame_height: Camera frame dimensions
        """
        # Flip X coordinate for mirror effect
        screen_x = (1 - hand_x) * self.screen_width
        screen_y = hand_y * self.screen_height
        
        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return (int(screen_x), int(screen_y))
    
    def move_cursor(self, hand_landmarks, frame_width, frame_height):
        """
        Move cursor based on index finger position
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_width, frame_height: Camera frame dimensions
        """
        index_tip = hand_landmarks[8]
        
        screen_pos = self.convert_to_screen_coords(
            index_tip.x, index_tip.y, frame_width, frame_height
        )
        
        # Apply smoothing
        smoothed_pos = self.smooth_position(screen_pos)
        
        # Move mouse cursor
        try:
            pyautogui.moveTo(smoothed_pos[0], smoothed_pos[1])
        except pyautogui.FailSafeException:
            print("Mouse moved to corner - failsafe activated")
    
    def detect_click_gesture(self, hand_landmarks):
        """
        Detect click gesture using thumb-index finger pinch
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            bool: True if click gesture detected
        """
        # Get thumb tip (4) and index finger tip (8)
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        
        # Calculate distance between thumb and index finger
        pinch_distance = self.get_pinch_distance(thumb_tip, index_tip)
        
        current_time = time.time()
        
        # Detect pinch (click)
        if pinch_distance < self.click_threshold:
            if not self.is_clicking:
                # Start of click gesture
                self.is_clicking = True
                self.click_start_time = current_time
                return False  # Don't click immediately
            else:
                # Continue clicking gesture
                if (current_time - self.click_start_time > self.click_duration_threshold and
                    current_time - self.last_click_time > self.click_cooldown):
                    # Valid click
                    self.last_click_time = current_time
                    return True
        else:
            # Release pinch
            self.is_clicking = False
        
        return False
    
    def detect_scroll_gesture(self, hand_landmarks):
        """
        Detect scroll gesture using middle finger movement
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (scroll_direction, scroll_amount) or (None, 0)
        """

        middle_tip = hand_landmarks[12]
        middle_mcp = hand_landmarks[9]  # Middle finger MCP joint
        
        # Calculate vertical distance between tip and MCP
        finger_extension = abs(middle_tip.y - middle_mcp.y)
        
        current_time = time.time()
        
        # Detect scroll gesture (extended middle finger)
        if finger_extension > self.scroll_threshold:
            if current_time - self.scroll_start_time > 0.2:  
                self.scroll_start_time = current_time
                
                # Determine scroll direction based on hand movement
                if self.prev_cursor_pos is not None:
                    current_pos = (middle_tip.x, middle_tip.y)
                    if len(self.cursor_history) > 3:
                        prev_pos = self.cursor_history[-3]
                        y_movement = current_pos[1] - prev_pos[1]
                        
                        if abs(y_movement) > 0.02:  # Minimum movement threshold
                            if y_movement > 0:
                                return ("down", 3)
                            else:
                                return ("up", 3)
        
        return (None, 0)
    
    def perform_click(self):
        """Perform a left mouse click"""
        try:
            pyautogui.click()
            print("Click performed")
        except Exception as e:
            print(f"Click failed: {e}")
    
    def perform_scroll(self, direction, amount):
        """
        Perform scroll action
        
        Args:
            direction: "up" or "down"
            amount: Scroll amount (positive integer)
        """
        try:
            if direction == "up":
                pyautogui.scroll(amount)
                print(f"Scrolled up {amount}")
            elif direction == "down":
                pyautogui.scroll(-amount)
                print(f"Scrolled down {amount}")
        except Exception as e:
            print(f"Scroll failed: {e}")
    
    def process_hand(self, hand_landmarks, frame_width, frame_height):
        """
        Process hand landmarks and perform mouse actions
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_width, frame_height: Camera frame dimensions
        """
        # Store current position in history
        index_tip = hand_landmarks[8]
        self.cursor_history.append((index_tip.x, index_tip.y))
        
        # Move cursor
        self.move_cursor(hand_landmarks, frame_width, frame_height)
        
        # Check for click gesture
        if self.detect_click_gesture(hand_landmarks):
            self.perform_click()
        
        # Check for scroll gesture
        scroll_direction, scroll_amount = self.detect_scroll_gesture(hand_landmarks)
        if scroll_direction:
            self.perform_scroll(scroll_direction, scroll_amount)
    
    def get_gesture_info(self, hand_landmarks):
        """
        Get information about current gestures for debugging
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            dict: Gesture information
        """
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        middle_mcp = hand_landmarks[9]
        
        pinch_distance = self.get_pinch_distance(thumb_tip, index_tip)
        finger_extension = abs(middle_tip.y - middle_mcp.y)
        
        return {
            "pinch_distance": pinch_distance,
            "finger_extension": finger_extension,
            "is_clicking": self.is_clicking,
            "click_threshold": self.click_threshold,
            "scroll_threshold": self.scroll_threshold
        }
