"""Player tracking implementation using ByteTrack algorithm."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Tracked object."""
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int = 0
    hits: int = 1
    
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PlayerTracker:
    """Simple IoU-based tracker (simplified ByteTrack)."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.frame_count = 0
        
    def update(self, detections: List) -> List[Track]:
        """Update tracks with new detections."""
        self.frame_count += 1
        
        if len(detections) == 0:
            # Age out tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].age += 1
                if self.tracks[track_id].age > self.max_age:
                    del self.tracks[track_id]
            return list(self.tracks.values())
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for i, det in enumerate(detections):
                self.tracks[self.next_id] = Track(
                    id=self.next_id,
                    bbox=det.bbox,
                    confidence=det.confidence
                )
                self.next_id += 1
            return list(self.tracks.values())
        
        # Simple distance-based matching
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()
        
        for track_id in track_ids:
            track = self.tracks[track_id]
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in matched_dets:
                    continue
                    
                iou = self._calculate_iou(track.bbox, det.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Update track
                det = detections[best_det_idx]
                self.tracks[track_id].bbox = det.bbox
                self.tracks[track_id].confidence = det.confidence
                self.tracks[track_id].age = 0
                self.tracks[track_id].hits += 1
                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)
            else:
                # Age track
                self.tracks[track_id].age += 1
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks[self.next_id] = Track(
                    id=self.next_id,
                    bbox=det.bbox,
                    confidence=det.confidence
                )
                self.next_id += 1
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].age > self.max_age:
                del self.tracks[track_id]
        
        return list(self.tracks.values())
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
