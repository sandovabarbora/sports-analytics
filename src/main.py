#!/usr/bin/env python3
"""Main entry point for Veo Analytics."""

import click
import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import time

@click.group()
def cli():
    """Veo Analytics CLI"""
    pass

@cli.command()
@click.argument('video_path')
@click.option('--output', '-o', default='outputs/analyzed.mp4')
@click.option('--model', '-m', default='yolov8n.pt')
@click.option('--report', is_flag=True)
def analyze(video_path, output, model, report):
    """Analyze video and save results."""
    print(f"Analyzing {video_path}")
    print(f"Model: {model}")
    
    # Load model
    yolo = YOLO(model)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output writer
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    print("Processing...")
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        results = yolo(frame, verbose=False)
        
        # Annotate
        annotated = results[0].plot()
        
        # Count detections
        if results[0].boxes is not None:
            total_detections += len(results[0].boxes)
        
        # Write frame
        out.write(annotated)
        
        frame_count += 1
        if frame_count % 60 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    if frame_count == 0:
        print("Error: No frames processed")
        return
        
    elapsed = time.time() - start_time
    
    print(f"\n✓ Analysis complete!")
    print(f"  Frames: {frame_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {frame_count/elapsed:.1f}")
    print(f"  Total detections: {total_detections}")
    print(f"  Output saved: {output}")
    
    if report:
        report_path = Path(output).with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write(f"Video Analysis Report\n")
            f.write(f"====================\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Frames: {frame_count}\n")
            f.write(f"Processing time: {elapsed:.2f}s\n")
            f.write(f"FPS: {frame_count/elapsed:.1f}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Avg detections/frame: {total_detections/max(frame_count, 1):.1f}\n")
        print(f"  Report saved: {report_path}")

@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8000)
def serve(host, port):
    """Start API server."""
    print(f"Starting API server on {host}:{port}")
    print("API server not implemented yet")

if __name__ == '__main__':
    cli()
