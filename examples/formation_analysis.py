import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

class FormationAnalyzer:
    """Analyze team formation and generate heatmaps."""
    
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.heatmap = np.zeros((720, 1280))
        self.all_positions = []  # Store all positions as flat list
        
    def process_video(self, video_path):
        """Process video and collect positions."""
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame, classes=[0])  # Only persons
            
            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1+x2)//2, (y1+y2)//2)
                    self.all_positions.append(center)
                    
                    # Update heatmap
                    if 0 <= center[0] < 1280 and 0 <= center[1] < 720:
                        cv2.circle(self.heatmap, center, 20, 1, -1)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
                
        cap.release()
        
        # Normalize heatmap
        self.heatmap = cv2.GaussianBlur(self.heatmap, (31, 31), 0)
        if self.heatmap.max() > 0:
            self.heatmap = self.heatmap / self.heatmap.max()
        
        print(f"Total positions collected: {len(self.all_positions)}")
        return self.heatmap
    
    def save_heatmap(self, output_path='outputs/heatmap.png'):
        """Save heatmap visualization."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(self.heatmap, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Player Density')
        plt.title('Player Position Heatmap')
        plt.xlabel('Field Width')
        plt.ylabel('Field Length')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap saved to {output_path}")
        
        # Save as colored overlay
        heatmap_colored = cv2.applyColorMap(
            (self.heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        overlay_path = output_path.replace('.png', '_overlay.png')
        cv2.imwrite(overlay_path, heatmap_colored)
        print(f"✓ Overlay saved to {overlay_path}")

# Generate heatmap
analyzer = FormationAnalyzer()
analyzer.process_video('data/samples/soccer/real_match.mp4')
analyzer.save_heatmap()

# Basic formation detection from average positions
if analyzer.all_positions:
    avg_x = np.mean([p[0] for p in analyzer.all_positions])
    avg_y = np.mean([p[1] for p in analyzer.all_positions])
    print(f"\nAverage position: ({avg_x:.0f}, {avg_y:.0f})")
    print(f"Most activity in: {'Left' if avg_x < 640 else 'Right'} side of field")
