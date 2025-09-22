import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict  # <-- Chybělo!
import matplotlib.pyplot as plt
from pathlib import Path

class PerformanceAnalyzer:
    """Calculate player performance metrics."""
    
    def __init__(self, video_path):
        self.model = YOLO('yolov8s.pt')
        self.video_path = video_path
        
        # Metrics storage
        self.player_distances = defaultdict(float)
        self.player_speeds = defaultdict(list)
        self.player_positions = defaultdict(list)
        
        # Field dimensions (VELMI hrubý odhad!)
        # Bez kalibrace kamery jsou to jen odhady
        self.pixel_to_meter = 100 / 1280  # Předpokládáme width = 100m
        
    def calculate_metrics(self):
        """Process video and calculate APPROXIMATE metrics."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        last_positions = {}
        frame_count = 0
        
        print("⚠️  UPOZORNĚNÍ: Metriky jsou pouze orientační!")
        print("   Pro přesné hodnoty je potřeba kalibrace kamery")
        print("="*50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame, classes=[0])
            
            if results[0].boxes:
                current_positions = {}
                
                for i, box in enumerate(results[0].boxes):
                    if i >= 22:  # Max 22 hráčů
                        break
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1+x2)/2, (y1+y2)/2)
                    player_id = f"P{i}"
                    
                    current_positions[player_id] = center
                    self.player_positions[player_id].append(center)
                    
                    # Calculate distance (APPROXIMATE)
                    if player_id in last_positions:
                        last_pos = last_positions[player_id]
                        
                        # Distance in pixels
                        dist_pixels = np.sqrt((center[0]-last_pos[0])**2 + 
                                            (center[1]-last_pos[1])**2)
                        
                        # Very rough conversion to meters
                        dist_meters = dist_pixels * self.pixel_to_meter
                        self.player_distances[player_id] += dist_meters
                        
                        # Speed estimate (probably overestimated)
                        speed = dist_meters * fps
                        self.player_speeds[player_id].append(speed)
                
                last_positions = current_positions
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Analyzed {frame_count} frames...")
        
        cap.release()
        
        # Summary
        duration_seconds = frame_count / fps
        
        print("\n" + "="*50)
        print("ORIENTAČNÍ METRIKY (nejsou přesné!)")
        print("="*50)
        print(f"Video Duration: {duration_seconds:.1f} seconds")
        print(f"Players Tracked: {len(self.player_distances)}")
        
        if self.player_distances:
            avg_distance = np.mean(list(self.player_distances.values()))
            print(f"\nPrůměrná vzdálenost na hráče: ~{avg_distance:.0f} metrů")
            print("(Reálně bude pravděpodobně 50-70% této hodnoty)")
        
        return self.player_distances, self.player_speeds

# Run analysis
analyzer = PerformanceAnalyzer('data/samples/soccer/real_match.mp4')
distances, speeds = analyzer.calculate_metrics()
