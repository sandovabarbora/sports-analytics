"""DTOs for match analysis."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class PlayerStatistics:
    """Statistics for a single player."""
    player_id: str
    team_id: str
    distance_covered: float
    avg_speed: float
    max_speed: float
    sprint_count: int
    positions: List[Tuple[float, float]]
    actions: Dict[str, int]


@dataclass
class TeamStatistics:
    """Statistics for a team."""
    team_id: str
    total_players: int
    formation: str
    possession_percentage: float
    avg_position: Tuple[float, float]
    player_stats: List[PlayerStatistics]


@dataclass
class MatchAnalysisDTO:
    """Complete match analysis results."""
    match_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    frames_processed: int
    team_stats: List[TeamStatistics]
    events: List[Dict[str, Any]]
    heatmap: Optional[np.ndarray] = None
