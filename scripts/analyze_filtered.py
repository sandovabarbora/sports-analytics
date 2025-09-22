from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')

# COCO classes:
# 0 = person
# 32 = sports ball
# 38 = tennis racket (chceme vyfiltrovat)

cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detekuj POUZE person a sports ball
    results = model(frame, classes=[0, 32], verbose=False)
    
    # Nebo filtruj po detekci
    # results = model(frame)
    # filtered_boxes = [box for box in results[0].boxes 
    #                   if box.cls in [0, 32]]
    
    annotated = results[0].plot()
    cv2.imshow('Filtered', annotated)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
