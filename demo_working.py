#!/usr/bin/env python3
"""Working demo for Veo Analytics."""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

def main():
    print("="*60)
    print("VEO ANALYTICS DEMO")
    print("="*60)
    
    # Check video exists
    video_path = Path("data/samples/soccer/synthetic_hd.mp4")
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    print(f"\n✓ Found video: {video_path}")
    print(f"  Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Load YOLO model
    print("\nLoading YOLO model...")
    model = YOLO('yolov8n.pt')  # nano for speed
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video loaded: {total_frames} frames @ {fps:.1f} FPS")
    print("\nProcessing... (Press Q to quit)")
    
    frame_count = 0
    start_time = time.time()
    fps_counter = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.perf_counter()
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Draw detections
        annotated = results[0].plot()
        
        # Calculate FPS
        frame_time = time.perf_counter() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_counter.append(current_fps)
        
        # Add info overlay
        cv2.rectangle(annotated, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(annotated, (0, 0), (300, 100), (255, 255, 255), 2)
        
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Count detections
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"Players: {num_detections}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Veo Analytics - Press Q to quit', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        # Progress every second
        if frame_count % int(fps) == 0:
            elapsed = time.time() - start_time
            avg_fps = np.mean(fps_counter[-int(fps):])
            print(f"Progress: {frame_count}/{total_frames} frames | "
                  f"Avg FPS: {avg_fps:.1f} | Time: {elapsed:.1f}s")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    total_time = time.time() - start_time
    avg_fps = np.mean(fps_counter)
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Min FPS: {min(fps_counter):.1f}")
    print(f"Max FPS: {max(fps_counter):.1f}")
    print("="*60)
    print("\n✓ Demo complete!")

if __name__ == "__main__":
    main()
