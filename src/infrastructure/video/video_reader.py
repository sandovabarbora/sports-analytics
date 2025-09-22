"""Video reading and processing infrastructure."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class VideoReader:
    """Video reader with frame iteration support."""
    
    def __init__(self):
        self.cap = None
        self.fps = 30
        self.frame_count = 0
        self.total_frames = 0
        self.start_time = None
        self.processing_time = 0
        
    def open(self, video_path: str) -> 'VideoReader':
        """Open video file or stream."""
        self.video_path = video_path
        
        # Handle different input types
        if video_path.isdigit():
            # Webcam
            self.cap = cv2.VideoCapture(int(video_path))
        elif video_path.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
            # Network stream
            self.cap = cv2.VideoCapture(video_path)
        else:
            # File
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Opened video: {video_path}")
        logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.total_frames}")
        
        return self
    
    def iter_frames(
        self, 
        start_time: float = 0, 
        end_time: float = -1,
        skip_frames: int = 0
    ) -> Iterator[np.ndarray]:
        """Iterate over video frames."""
        if not self.cap:
            raise RuntimeError("Video not opened")
        
        # Set start position
        if start_time > 0:
            start_frame = int(start_time * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.frame_count = start_frame
        
        # Calculate end frame
        end_frame = self.total_frames
        if end_time > 0:
            end_frame = min(int(end_time * self.fps), self.total_frames)
        
        self.start_time = datetime.now()
        frames_processed = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or (end_time > 0 and self.frame_count >= end_frame):
                break
            
            # Skip frames if requested
            if skip_frames > 0 and frames_processed % (skip_frames + 1) != 0:
                frames_processed += 1
                self.frame_count += 1
                continue
            
            self.frame_count += 1
            frames_processed += 1
            
            yield frame
        
        self.processing_time = (datetime.now() - self.start_time).total_seconds()
    
    @property
    def current_timestamp(self) -> datetime:
        """Get current frame timestamp."""
        if self.start_time is None:
            self.start_time = datetime.now()
        
        frame_time = self.frame_count / self.fps
        return self.start_time + timedelta(seconds=frame_time)
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    """Video writer for saving processed videos."""
    
    def __init__(
        self, 
        output_path: str,
        fps: float = 30,
        width: int = 1280,
        height: int = 720,
        codec: str = 'mp4v'
    ):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")
        
        logger.info(f"Created video writer: {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write frame to video."""
        # Resize if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
