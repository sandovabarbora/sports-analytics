"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile


@pytest.fixture
def sample_frame():
    """Generate sample video frame."""
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    # Add some "players"
    cv2.circle(frame, (300, 400), 20, (255, 0, 0), -1)
    cv2.circle(frame, (600, 400), 20, (0, 0, 255), -1)
    return frame


@pytest.fixture
def sample_video(tmp_path):
    """Generate sample video file."""
    video_path = tmp_path / "test_video.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30, (1280, 720))
    
    for i in range(30):  # 1 second of video
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        cv2.circle(frame, (300 + i*10, 400), 20, (255, 0, 0), -1)
        cv2.circle(frame, (600 - i*10, 400), 20, (0, 0, 255), -1)
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def mock_detector():
    """Mock detector for testing."""
    from src.infrastructure.ml.detector import IPlayerDetector, Detection
    
    class MockDetector(IPlayerDetector):
        def detect(self, frame):
            return [
                Detection(bbox=(100, 100, 150, 200), confidence=0.9, class_id=0),
                Detection(bbox=(300, 300, 350, 400), confidence=0.8, class_id=0)
            ]
    
    return MockDetector()


@pytest.fixture
def mock_tracker():
    """Mock tracker for testing."""
    from src.infrastructure.ml.tracker import PlayerTracker
    return PlayerTracker()


@pytest.fixture
def mock_video_reader():
    """Mock video reader for testing."""
    from src.infrastructure.video.video_reader import VideoReader
    return VideoReader()
