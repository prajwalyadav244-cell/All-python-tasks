# pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt') # pre-trained model

cap = cv2.VideoCapture(0) # 0 for webcam or give 'video.mp4'

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Process each video frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    # Display output with labels and tracking IDs
    annotated_frame = results[0].plot()
    cv2.imshow("CodeAlpha - Object Detection & Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
