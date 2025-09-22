import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

class PassAnalyzer:
    """Detect passes between players and build passing network."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.ball_owner = None
        self.last_ball_owner = None
        self.passes = defaultdict(lambda: defaultdict(int))  # from_player -> to_player -> count
        self.ball_history = []
        
    def find_closest_player_to_ball(self, players, ball_pos):
        """Find which player has the ball."""
        if not players or not ball_pos:
            return None
            
        min_dist = float('inf')
        closest_player = None
        
        for player_id, bbox in players.items():
            x1, y1, x2, y2 = bbox
            player_center = ((x1+x2)/2, (y1+y2)/2)
            dist = math.sqrt((player_center[0]-ball_pos[0])**2 + 
                           (player_center[1]-ball_pos[1])**2)
            
            if dist < min_dist and dist < 50:  # Within 50 pixels
                min_dist = dist
                closest_player = player_id
                
        return closest_player
    
    def process_frame(self, frame):
        """Detect passes in frame."""
        results = self.model(frame, classes=[0, 32])  # person, sports ball
        
        # Get players and ball
        players = {}
        ball_pos = None
        
        for box in results[0].boxes:
            if box.cls == 0:  # Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Simple ID based on position (in production use proper tracking)
                player_id = f"{x1//100}_{y1//100}"
                players[player_id] = (x1, y1, x2, y2)
                
            elif box.cls == 32:  # Ball
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_pos = ((x1+x2)//2, (y1+y2)//2)
        
        # Detect ball possession changes
        if ball_pos:
            current_owner = self.find_closest_player_to_ball(players, ball_pos)
            
            if current_owner != self.ball_owner:
                if self.ball_owner and current_owner:
                    # Pass detected!
                    self.passes[self.ball_owner][current_owner] += 1
                    cv2.putText(frame, f"PASS: {self.ball_owner} -> {current_owner}", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.last_ball_owner = self.ball_owner
                self.ball_owner = current_owner
        
        # Draw current possession
        if self.ball_owner and self.ball_owner in players:
            x1, y1, x2, y2 = players[self.ball_owner]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, "HAS BALL", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show pass statistics
        total_passes = sum(sum(to_dict.values()) for to_dict in self.passes.values())
        cv2.putText(frame, f"Total Passes: {total_passes}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def get_pass_network(self):
        """Get passing network statistics."""
        return dict(self.passes)

# Run pass analysis
analyzer = PassAnalyzer()
cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    result = analyzer.process_frame(frame)
    cv2.imshow('Pass Analysis', result)
    
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nPass Network:")
for from_player, to_dict in analyzer.passes.items():
    for to_player, count in to_dict.items():
        print(f"  {from_player} -> {to_player}: {count} passes")
