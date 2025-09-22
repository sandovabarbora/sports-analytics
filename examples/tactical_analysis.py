import cv2
import numpy as np
from ultralytics import YOLO

class TacticalBoard:
    """Create top-down 2D tactical view."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.board_width = 400
        self.board_height = 600
        
    def create_tactical_board(self):
        """Create empty tactical board."""
        board = np.ones((self.board_height, self.board_width, 3), dtype=np.uint8) * 50
        
        # Draw field lines
        cv2.rectangle(board, (20, 20), (self.board_width-20, self.board_height-20), 
                     (255, 255, 255), 2)
        
        # Center line
        cv2.line(board, (20, self.board_height//2), 
                (self.board_width-20, self.board_height//2), (255, 255, 255), 2)
        
        # Center circle
        cv2.circle(board, (self.board_width//2, self.board_height//2), 40, 
                  (255, 255, 255), 2)
        
        # Goal areas
        cv2.rectangle(board, (self.board_width//2-60, 20), 
                     (self.board_width//2+60, 80), (255, 255, 255), 2)
        cv2.rectangle(board, (self.board_width//2-60, self.board_height-80), 
                     (self.board_width//2+60, self.board_height-20), (255, 255, 255), 2)
        
        return board
    
    def map_to_board(self, x, y, frame_width, frame_height):
        """Map frame coordinates to board coordinates."""
        board_x = int(x * self.board_width / frame_width)
        board_y = int(y * self.board_height / frame_height)
        return board_x, board_y
    
    def process_frame(self, frame):
        """Create tactical view from frame."""
        h, w = frame.shape[:2]
        board = self.create_tactical_board()
        
        results = self.model(frame, classes=[0, 32])
        
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Map to board
                board_x, board_y = self.map_to_board(center_x, center_y, w, h)
                
                if box.cls == 0:  # Person
                    # Simple team classification by position
                    color = (0, 0, 255) if center_y < h//2 else (255, 0, 0)
                    cv2.circle(board, (board_x, board_y), 8, color, -1)
                    cv2.circle(board, (board_x, board_y), 8, (255, 255, 255), 2)
                    
                elif box.cls == 32:  # Ball
                    cv2.circle(board, (board_x, board_y), 5, (255, 255, 0), -1)
        
        # Combine views
        board_resized = cv2.resize(board, (w//3, h//3))
        frame[10:10+h//3, w-w//3-10:w-10] = board_resized
        
        return frame

# Run tactical view
tactical = TacticalBoard()
cap = cv2.VideoCapture('data/samples/soccer/real_match.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    result = tactical.process_frame(frame)
    cv2.imshow('Tactical View', result)
    
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
