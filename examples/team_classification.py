import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

class TeamClassifier:
    """Classify players into teams based on jersey color."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.team_colors = {}
        
    def extract_jersey_color(self, frame, bbox):
        """Extract dominant color from upper body (jersey area)."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get upper 40% of bbox (jersey area)
        jersey_y2 = y1 + int((y2 - y1) * 0.4)
        jersey_region = frame[y1:jersey_y2, x1:x2]
        
        if jersey_region.size == 0:
            return None
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Get dominant color (excluding white/green - field/lines)
        mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
        mean_color = cv2.mean(hsv, mask=mask)[:3]
        
        return mean_color
    
    def classify_teams(self, frame):
        """Detect players and classify into teams."""
        results = self.model(frame, classes=[0])  # Only persons
        
        if not results[0].boxes:
            return frame
            
        # Extract colors for all players
        colors = []
        valid_boxes = []
        
        for box in results[0].boxes:
            color = self.extract_jersey_color(frame, box.xyxy[0])
            if color:
                colors.append(color)
                valid_boxes.append(box)
        
        if len(colors) < 2:
            return frame
            
        # Cluster into 2 teams (+ goalkeeper = 3 clusters)
        kmeans = KMeans(n_clusters=min(3, len(colors)))
        labels = kmeans.fit_predict(colors)
        
        # Draw boxes with team colors
        team_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Red, Blue, Green
        
        for box, label in zip(valid_boxes, labels):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = team_colors[label]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Team {label+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return frame

# Test it
classifier = TeamClassifier()
cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    result = classifier.classify_teams(frame)
    cv2.imshow('Team Classification', result)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
