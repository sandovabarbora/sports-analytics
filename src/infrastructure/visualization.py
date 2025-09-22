"""Visualization utilities for analytics results."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import seaborn as sns


def draw_tracks(frame: np.ndarray, tracks: List[Any]) -> np.ndarray:
    """Draw tracks on frame."""
    annotated = frame.copy()
    
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        
        # Draw bbox
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID
        label = f"ID: {track.id}"
        cv2.putText(annotated, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated


def generate_heatmap(positions: List[Tuple[float, float]], 
                    width: int = 1280, 
                    height: int = 720) -> np.ndarray:
    """Generate position heatmap."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for x, y in positions:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            # Add gaussian blob
            cv2.circle(heatmap, (x, y), 30, 1.0, -1)
    
    # Apply gaussian blur for smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (61, 61), 0)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), 
                                        cv2.COLORMAP_JET)
    
    return heatmap_colored


def plot_statistics(stats: Dict[str, Any], save_path: str = None):
    """Plot match statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Players detected over time
    if 'players_per_frame' in stats:
        axes[0, 0].plot(stats['players_per_frame'])
        axes[0, 0].set_title('Players Detected')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Count')
    
    # Speed distribution
    if 'speed_distribution' in stats:
        axes[0, 1].hist(stats['speed_distribution'], bins=20)
        axes[0, 1].set_title('Speed Distribution')
        axes[0, 1].set_xlabel('Speed (m/s)')
        axes[0, 1].set_ylabel('Frequency')
    
    # Distance covered
    if 'distance_per_player' in stats:
        axes[1, 0].bar(range(len(stats['distance_per_player'])), 
                      stats['distance_per_player'])
        axes[1, 0].set_title('Distance Covered')
        axes[1, 0].set_xlabel('Player ID')
        axes[1, 0].set_ylabel('Distance (m)')
    
    # Actions
    if 'action_counts' in stats:
        actions = list(stats['action_counts'].keys())
        counts = list(stats['action_counts'].values())
        axes[1, 1].pie(counts, labels=actions, autopct='%1.1f%%')
        axes[1, 1].set_title('Action Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return fig
