import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class EventDetector:
    """Detect important events: goals, corners, throw-ins."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.ball_positions = deque(maxlen=30)
        self.events = []
        
    def detect_ball_out_of_bounds(self, ball_pos, frame_shape):
        """Check if ball went out of bounds."""
        h, w = frame_shape[:2]
        x, y = ball_pos
        
        margin = 50
        if x < margin or x > w-margin or y < margin or y > h-margin:
            return True
        return False
    
    def detect_goal_opportunity(self, ball_pos, frame_shape):
        """Detect if ball is near goal area."""
        h, w = frame_shape[:2]
        x, y = ball_pos
        
        # Goal areas (simplified)
        left_goal = x < w*0.1 and h*0.3 < y < h*0.7
        right_goal = x > w*0.9 and h*0.3 < y < h*0.7
        
        return left_goal or right_goal
    
    def process_frame(self, frame, frame_num):
        """Detect events in frame."""
        results = self.model(frame, classes=[0, 32])  # person, sports ball
        
        # Find ball
        ball_detected = False
        for box in results[0].boxes:
            if box.cls == 32:  # sports ball
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_pos = ((x1+x2)//2, (y1+y2)//2)
                self.ball_positions.append(ball_pos)
                ball_detected = True
                
                # Check events
                if self.detect_ball_out_of_bounds(ball_pos, frame.shape):
                    event = f"Frame {frame_num}: Ball out of bounds!"
                    self.events.append(event)
                    cv2.putText(frame, "OUT OF BOUNDS!", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                if self.detect_goal_opportunity(ball_pos, frame.shape):
                    event = f"Frame {frame_num}: Goal opportunity!"
                    self.events.append(event)
                    cv2.putText(frame, "GOAL CHANCE!", (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Draw ball trail
                if len(self.ball_positions) > 1:
                    points = np.array(list(self.ball_positions), np.int32)
                    cv2.polylines(frame, [points], False, (0, 255, 255), 3)
        
        return frame

# Run event detection
detector = EventDetector()
cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    result = detector.process_frame(frame, frame_num)
    cv2.imshow('Event Detection', result)
    
    if cv2.waitKey(30) == ord('q'):
        break
        
    frame_num += 1

cap.release()
cv2.destroyAllWindows()

print("\nDetected Events:")
for event in detector.events:
    print(f"  {event}")
