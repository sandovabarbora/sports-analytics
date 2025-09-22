#!/usr/bin/env python3
"""Simple demo script to test the system."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure.ml.detector import YOLOPlayerDetector
from src.infrastructure.ml.tracker import PlayerTracker
from src.infrastructure.video.video_reader import VideoReader
from src.infrastructure.visualization import draw_tracks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run simple demo."""
    video_path = "data/samples/soccer/synthetic_test.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        logger.info("Please run setup.sh first to create sample videos")
        return
    
    logger.info("Initializing components...")
    detector = YOLOPlayerDetector()
    tracker = PlayerTracker()
    reader = VideoReader()
    
    logger.info(f"Processing video: {video_path}")
    reader.open(video_path)
    
    frame_count = 0
    for frame in reader.iter_frames(end_time=5.0):  # Process first 5 seconds
        # Detect
        detections = detector.detect(frame)
        
        # Track
        tracks = tracker.update(detections)
        
        # Visualize
        annotated = draw_tracks(frame, tracks)
        
        # Log progress
        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count} frames, {len(tracks)} players tracked")
        
        # Display (optional - comment out for headless)
        # import cv2
        # cv2.imshow('Demo', annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    reader.release()
    logger.info(f"Demo complete! Processed {frame_count} frames")


if __name__ == "__main__":
    main()
