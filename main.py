from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLO model
model = YOLO("yolo11n-seg.pt")

# Open video capture
cap = cv2.VideoCapture(0)

# Load background images
background_folder = "backgrounds"
background_files = sorted([os.path.join(background_folder, f) for f in os.listdir(background_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
background_index = 0

# Flag to show/hide background
show_background = True

def toggle_background():
    global show_background
    show_background = not show_background

def next_background():
    global background_index
    background_index = (background_index + 1) % len(background_files)

def previous_background():
    global background_index
    background_index = (background_index - 1) % len(background_files)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Load and resize background
    background_img = cv2.imread(background_files[background_index])
    background_img_resized = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))

    # Get results from YOLO segmentation
    results = model(frame)
    mask_overlay = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for result in results:
        for mask in result.masks.xy:
            mask = np.array(mask, dtype=np.int32)
            cv2.fillPoly(mask_overlay, [mask], 255)
    
    # Convert mask to 3 channels
    mask_3ch = cv2.merge([mask_overlay, mask_overlay, mask_overlay])
    
    # Extract foreground
    foreground = cv2.bitwise_and(frame, mask_3ch)
    
    # Combine foreground with background image if enabled
    if show_background:
        output = np.where(mask_3ch == 255, foreground, background_img_resized)
    else:
        output = frame

    # Show the instructions
    cv2.putText(output, "Press T to toggle background", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(output, "Press left/right arrow keys to change background", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLO Background Removal", output)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("t"):
        toggle_background()
    elif key == 81:  # Left arrow key
        previous_background()
    elif key == 83:  # Right arrow key
        next_background()

cap.release()
cv2.destroyAllWindows()
