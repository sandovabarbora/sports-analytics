import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

class PlayerTracker:
    """Track players without track() method - use simple IoU matching."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def process_frame(self, frame):
        """Process frame with simple tracking."""
        results = self.model(frame, classes=[0])  # Only persons
        
        if results[0].boxes is None:
            return frame
            
        current_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_boxes.append((x1, y1, x2, y2))
        
        # Simple matching with previous frame
        matched = {}
        for curr_idx, curr_box in enumerate(current_boxes):
            best_iou = 0
            best_id = None
            
            for track_id, track_data in self.tracks.items():
                if 'last_box' in track_data:
                    iou = self.calculate_iou(curr_box, track_data['last_box'])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_id = track_id
            
            if best_id is not None:
                matched[curr_idx] = best_id
            else:
                matched[curr_idx] = self.next_id
                self.tracks[self.next_id] = {'positions': []}
                self.next_id += 1
        
        # Update tracks and draw
        for curr_idx, track_id in matched.items():
            x1, y1, x2, y2 = current_boxes[curr_idx]
            center = ((x1+x2)//2, (y1+y2)//2)
            
            self.tracks[track_id]['last_box'] = current_boxes[curr_idx]
            self.tracks[track_id]['positions'].append(center)
            
            # Keep only last 30 positions
            if len(self.tracks[track_id]['positions']) > 30:
                self.tracks[track_id]['positions'].pop(0)
            
            # Calculate speed
            speed = 0
            if len(self.tracks[track_id]['positions']) > 1:
                p1 = self.tracks[track_id]['positions'][-2]
                p2 = self.tracks[track_id]['positions'][-1]
                distance = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                speed = distance * 0.5  # Rough estimate
            
            # Draw box
            color = (0, 255, 0) if speed < 5 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"ID:{track_id} | {speed:.1f} m/s"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw trail
            if len(self.tracks[track_id]['positions']) > 1:
                points = np.array(self.tracks[track_id]['positions'], np.int32)
                cv2.polylines(frame, [points], False, (255, 255, 0), 2)
        
        self.frame_count += 1
        return frame

# Run tracking
tracker = PlayerTracker()
cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    result = tracker.process_frame(frame)
    
    # Add stats
    cv2.putText(result, f"Tracking {len(tracker.tracks)} players", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Player Tracking', result)
    
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTracked {len(tracker.tracks)} unique players")
