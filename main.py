import threading
from ultralytics import YOLO
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import keyboard
import math
import mss
import time

# Load the YOLO model
model = YOLO(r"model\model_- 31 august 2024 2_41.onnx")

# Global variables for sharing data between threads
me_position = None
nearest_enemy_position = None
enemies = []
frame = None
lock = threading.Lock()

# Function to get the browser window
def get_browser_window():
    browser_titles = ['Chrome', 'Firefox', 'Edge', 'Safari', 'Opera']
    for title in browser_titles:
        windows = gw.getWindowsWithTitle(title)
        if windows:
            return windows[0]
    return None

# Function to find the nearest enemy
def find_nearest_enemy(me_position, enemies):
    if not enemies:
        return None
    nearest_enemy = min(enemies, key=lambda e: math.dist(me_position, e))
    return nearest_enemy

# Thread to capture the screen
def capture_screen(browser_window):
    global frame
    with mss.mss() as sct:
        while True:
            monitor = {
                "top": browser_window.top, 
                "left": browser_window.left, 
                "width": browser_window.width, 
                "height": browser_window.height
            }
            screenshot = sct.grab(monitor)

            # Lock access to shared frame variable
            with lock:
                # Resize the captured frame for faster processing
                frame = cv2.resize(np.array(screenshot), (monitor["width"]//2, monitor["height"]//2))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Thread to run YOLO detection
def run_yolo_detection():
    global me_position, nearest_enemy_position, enemies, frame
    while True:
        if frame is not None:
            # Lock access to shared frame variable
            with lock:
                local_frame = frame.copy()

            # Run YOLO detection
            results = model(local_frame, imgsz=640, conf=0.005)  # Adjust imgsz for faster processing

            # Clear previous detections
            me_position = None
            enemies.clear()

            # Process the results
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    confidence = box.conf[0]  # Get the confidence score
                    
                    # Skip results with confidence lower than 0.15
                    if confidence < 0.15:
                        continue
                    
                    # Calculate box area and frame area
                    box_area = (x2 - x1) * (y2 - y1)
                    frame_area = local_frame.shape[0] * local_frame.shape[1]
                    
                    # Filter out boxes that are too large
                    if box_area / frame_area > 0.005:
                        continue

                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    if class_name == 'Me':
                        me_position = (center_x, center_y)
                        color = (0, 255, 0)  # Green for Me
                    elif class_name == 'Emeny':
                        enemies.append(((x1, y1, x2, y2), (center_x, center_y)))
                        color = (0, 0, 255)  # Red for Enemy

                    # Draw the bounding box and label
                    cv2.rectangle(local_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(local_frame, f"{class_name} {confidence:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # After processing, update nearest enemy
            if me_position and enemies:
                nearest_enemy = find_nearest_enemy(me_position, [e[1] for e in enemies])
                nearest_enemy_position = nearest_enemy
                
                # Draw blue box around the nearest enemy
                for enemy in enemies:
                    if enemy[1] == nearest_enemy:
                        x1, y1, x2, y2 = enemy[0]
                        cv2.rectangle(local_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for nearest enemy
                        break

            # Lock access to shared frame variable
            with lock:
                frame = local_frame

        time.sleep(0.001)  # Adjust the sleep time for better performance

# Thread to handle user input and actions
def handle_user_input(browser_window):
    global me_position, nearest_enemy_position
    while True:
        # Check if 'k' key is pressed
        if keyboard.is_pressed('k'):
            # Get the most recent me_position and nearest_enemy_position
            with lock:
                current_me_position = me_position
                current_nearest_enemy_position = nearest_enemy_position
            
            if current_me_position and current_nearest_enemy_position:
                # Calculate the midpoint between "Me" and the nearest enemy
                midpoint_x = (current_me_position[0] + current_nearest_enemy_position[0]) // 2
                midpoint_y = (current_me_position[1] + current_nearest_enemy_position[1]) // 2

                # Move mouse to the midpoint and click
                pyautogui.moveTo(browser_window.left + midpoint_x * 2,  # Adjust for resized frame
                                 browser_window.top + midpoint_y * 2, duration=0)
                pyautogui.click()

        time.sleep(0.001)  # Adjust the sleep time for better performance

# Get the browser window
browser_window = get_browser_window()
if browser_window is None:
    print("No browser window found. Please open a browser and try again.")
    exit()

# Start the threads
t1 = threading.Thread(target=capture_screen, args=(browser_window,))
t2 = threading.Thread(target=run_yolo_detection)
t3 = threading.Thread(target=handle_user_input, args=(browser_window,))

t1.start()
t2.start()
t3.start()

# Main thread for displaying the result
while True:
    if frame is not None:
        with lock:
            cv2.imshow("YOLO Real-time Browser Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

# Wait for threads to finish
t1.join()
t2.join()
t3.join()
