#!/usr/bin/env python3
"""Optimized demo using UV environment."""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

# Check if MPS is available (Apple Silicon)
if torch.backends.mps.is_available():
    device = 'mps'
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = 'cuda'
    print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name()}")
else:
    device = 'cpu'
    print("⚠ Using CPU (slower)")

def run_demo():
    """Run optimized demo."""
    # Load model
    print("\nLoading YOLO model...")
    model = YOLO('yolov8n.pt')
    model.to(device)
    
    # Open video
    video_path = Path('data/samples/soccer/synthetic_hd.mp4')
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Process video
    print(f"\nProcessing video at {fps:.1f} FPS...")
    frame_times = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.perf_counter()
        
        # Run inference
        results = model(frame, device=device, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot(
            conf=True,
            labels=True,
            boxes=True,
            masks=False,
            probs=False
        )
        
        # Calculate FPS
        inference_time = time.perf_counter() - start_time
        frame_times.append(inference_time)
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Add FPS counter
        cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add device info
        cv2.putText(annotated_frame, f'Device: {device.upper()}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display
        cv2.imshow('Veo Analytics Demo (Press Q to quit)', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        # Print progress every second
        if frame_count % int(fps) == 0:
            avg_fps = 1.0 / np.mean(frame_times[-int(fps):])
            print(f"Processed {frame_count} frames | Avg FPS: {avg_fps:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*50)
    print("PERFORMANCE STATISTICS")
    print("="*50)
    print(f"Total frames: {frame_count}")
    print(f"Avg FPS: {1.0/np.mean(frame_times):.1f}")
    print(f"Min FPS: {1.0/np.max(frame_times):.1f}")
    print(f"Max FPS: {1.0/np.min(frame_times):.1f}")
    print(f"Device: {device.upper()}")
    print("="*50)

if __name__ == "__main__":
    run_demo()
