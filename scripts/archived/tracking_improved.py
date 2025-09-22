import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

class ImprovedTracker:
    """Better tracking with max 22 players constraint."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.tracks = {}
        self.next_id = 0
        self.max_players = 22  # Football constraint!
        
    def process_frame(self, frame):
        """Process with player limit."""
        results = self.model(frame, classes=[0])  # persons
        
        if results[0].boxes is None:
            return frame
            
        # Sort by confidence - keep best detections
        boxes_conf = [(box.xyxy[0], box.conf[0]) for box in results[0].boxes]
        boxes_conf.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top 22 detections (max players on field)
        boxes_conf = boxes_conf[:self.max_players]
        
        # Simple tracking with constraint
        active_tracks = []
        for box, conf in boxes_conf:
            x1, y1, x2, y2 = map(int, box)
            
            # Find closest existing track
            best_dist = float('inf')
            best_id = None
            
            for tid, track in self.tracks.items():
                if 'last_pos' in track:
                    dist = math.sqrt(
                        ((x1+x2)/2 - track['last_pos'][0])**2 + 
                        ((y1+y2)/2 - track['last_pos'][1])**2
                    )
                    if dist < best_dist and dist < 100:  # Max 100px movement
                        best_dist = dist
                        best_id = tid
            
            # Use existing or create new (if under limit)
            if best_id is not None:
                track_id = best_id
            elif len(self.tracks) < self.max_players:
                track_id = self.next_id
                self.tracks[track_id] = {'positions': []}
                self.next_id += 1
            else:
                continue  # Skip if at max players
            
            # Update track
            center = ((x1+x2)//2, (y1+y2)//2)
            self.tracks[track_id]['last_pos'] = center
            self.tracks[track_id]['positions'].append(center)
            active_tracks.append(track_id)
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"P{track_id}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Clean old tracks
        for tid in list(self.tracks.keys()):
            if tid not in active_tracks:
                if 'positions' in self.tracks[tid]:
                    if len(self.tracks[tid]['positions']) < 10:
                        del self.tracks[tid]  # Remove if seen briefly
        
        # Stats
        cv2.putText(frame, f"Active Players: {len(active_tracks)} / Max: {self.max_players}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

# Run improved tracking
tracker = ImprovedTracker()
cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    result = tracker.process_frame(frame)
    cv2.imshow('Improved Tracking', result)
    
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTotal unique players seen: {len(tracker.tracks)}")
print(f"Expected: ≤22 players")
