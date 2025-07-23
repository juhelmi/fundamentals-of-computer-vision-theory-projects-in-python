import cv2
import numpy as np
import os

# install also opencv-contrib-python
# Model files from https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

# --- Step 1: Download Nano tracker ONNX models ---
# (Assume you have 'nanotrack_backbone.onnx' and 'nanotrack_neckhead.onnx'
# in your project directory)

# --- Step 2: Initialize your video source (e.g., webcam or video file) ---
cap = cv2.VideoCapture(0) # For webcam
if not cap.isOpened():
    print(f"Ei aukee")
    exit(1)
# cap = cv2.VideoCapture('your_video.mp4') # For video file

# --- Step 3: Initial Object Detection (using a placeholder here) ---
# In a real application, you'd use a DNN model (YOLO, SSD) here.
# For simplicity, we'll let the user select the ROI manually in the first frame.

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video frame.")
    exit()

# Select ROI (Region of Interest) for the object to track
# The user will draw a bounding box manually
bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object to Track")

# Initialize the Nano tracker
try:
    # TrackerNano requires model paths.
    # Replace with the actual paths to your downloaded ONNX files.
    params = cv2.TrackerNano_Params()
    # params.backbone = "data/nanotrack_backbone.onnx"
    # params.neckhead = "data/nanotrack_neckhead.onnx"
    params.backbone = "data/nanotrack_backbone_sim.onnx"
    params.neckhead = "data/nanotrack_head_sim.onnx"
    if not os.path.exists(params.backbone) or not os.path.exists(params.neckhead):
        print(f"Model files are not found, {params.neckhead} {params.neckhead}")
        exit(2)
    #tracker = cv2.TrackerNano.create(params)
    tracker = cv2.TrackerNano.create(parameters=params)
    tracker.init(frame, bbox)
    print(f"Tracker init might work")
except Exception as e:
    print(f"Error initializing TrackerNano. Make sure you have opencv-contrib-python installed and the ONNX models are correctly specified: {e}")
    exit()

print("TrackerNano initialized successfully.")

# --- Step 4: Tracking loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    if success:
        # Tracking successful, draw bounding box
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Tracking failed
        cv2.putText(frame, "Tracking Lost!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Object Tracking with Nano", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
