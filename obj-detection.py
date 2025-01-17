import cv2
import torch
import numpy as np

# Load YOLOv5 model (change 'yolov5m' to 'yolov5l' for even higher accuracy)
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get the bounding boxes, labels, and confidence scores
    labels, coords, confs = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1].numpy(), results.xyxyn[0][:, -2]

    # Draw bounding boxes on the frame
    for label, coord, conf in zip(labels, coords, confs):
        x1, y1, x2, y2 = int(coord[0] * frame.shape[1]), int(coord[1] * frame.shape[0]), int(coord[2] * frame.shape[1]), int(coord[3] * frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{model.names[int(label)]} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if the space key is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
