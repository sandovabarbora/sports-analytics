"""Performance benchmarking utilities."""

import time
import numpy as np
from typing import Dict, Any
import psutil
import GPUtil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_benchmark(video_path: str, num_frames: int = 100) -> Dict[str, Any]:
    """Run performance benchmark on video processing."""
    from src.infrastructure.ml.detector import YOLOPlayerDetector
    from src.infrastructure.ml.tracker import PlayerTracker
    from src.infrastructure.video.video_reader import VideoReader
    
    results = {
        'detection_fps': 0,
        'tracking_fps': 0,
        'total_fps': 0,
        'memory_usage_mb': 0,
        'gpu_usage_percent': 0,
        'cpu_usage_percent': 0
    }
    
    try:
        # Initialize components
        detector = YOLOPlayerDetector()
        tracker = PlayerTracker()
        reader = VideoReader()
        
        # Open video
        reader.open(video_path)
        
        # Warmup
        for i, frame in enumerate(reader.iter_frames()):
            if i >= 10:
                break
            _ = detector.detect(frame)
        
        # Reset reader
        reader.release()
        reader.open(video_path)
        
        # Benchmark
        detection_times = []
        tracking_times = []
        total_times = []
        
        for i, frame in enumerate(reader.iter_frames()):
            if i >= num_frames:
                break
            
            # Total time
            start_total = time.perf_counter()
            
            # Detection
            start = time.perf_counter()
            detections = detector.detect(frame)
            detection_time = time.perf_counter() - start
            detection_times.append(detection_time)
            
            # Tracking
            start = time.perf_counter()
            tracks = tracker.update(detections)
            tracking_time = time.perf_counter() - start
            tracking_times.append(tracking_time)
            
            total_time = time.perf_counter() - start_total
            total_times.append(total_time)
        
        # Calculate metrics
        results['detection_fps'] = 1.0 / np.mean(detection_times)
        results['tracking_fps'] = 1.0 / np.mean(tracking_times)
        results['total_fps'] = 1.0 / np.mean(total_times)
        
        # System metrics
        process = psutil.Process()
        results['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        results['cpu_usage_percent'] = process.cpu_percent()
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                results['gpu_usage_percent'] = gpus[0].load * 100
                results['gpu_memory_mb'] = gpus[0].memoryUsed
        except:
            pass
        
        logger.info(f"Benchmark results: {results}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
    
    finally:
        reader.release()
    
    return results


def profile_memory(func):
    """Decorator to profile memory usage."""
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        logger.info(f"{func.__name__} - Memory: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
        
        return result
    return wrapper
