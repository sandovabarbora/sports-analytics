# Sports Analytics - Real-time Video Analysis System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive computer vision system for real-time sports video analysis, focusing on player tracking, performance metrics, and tactical insights.

## 🎯 Features

### Core Capabilities
- **Multi-Object Detection & Tracking** - Real-time player and ball detection using YOLOv8
- **Team Classification** - Automatic team identification based on jersey colors
- **Performance Metrics** - Speed, distance, and intensity zone analysis
- **Event Detection** - Automatic detection of key events (ball out of bounds, goal opportunities)
- **Tactical Analysis** - Formation detection and player position heatmaps
- **Live Visualization** - Real-time tactical board and player trails

### Visual Analytics
- Position heatmaps showing player movement patterns
- Tactical board with 2D bird's-eye view
- Player trajectories and speed visualization
- Pass network analysis
- Ball trajectory tracking

## 📊 Performance

| Model    | FPS (CPU) | FPS (GPU) | mAP  | Use Case |
|----------|-----------|-----------|------|----------|
| YOLOv8n  | 30.4      | 100+      | 0.37 | Real-time analysis |
| YOLOv8s  | 16.2      | 60+       | 0.45 | Balanced performance |
| YOLOv8m  | 7.5       | 35+       | 0.50 | Maximum accuracy |

*Benchmarked on MacBook Pro M1 (CPU/MPS) and NVIDIA RTX 3080 (GPU)*

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- ffmpeg
- 4GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sports-analytics.git
cd sports-analytics

# Option 1: Using UV (recommended - faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Option 2: Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download YOLO models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Basic Usage

```python
# Quick demo
python demo_working.py

# Analyze your own video
python src/main.py analyze path/to/your/video.mp4 \
    --output results/analyzed.mp4 \
    --model yolov8s.pt \
    --report

# Start API server
python src/main.py serve
# Visit http://localhost:8000/docs for API documentation
```

## 📁 Project Structure

```
sports-analytics/
├── src/
│   ├── domain/          # Core business logic
│   │   ├── entities/    # Player, Team, Match entities
│   │   └── services/    # Business rules
│   ├── application/     # Use cases
│   │   ├── use_cases/   # AnalyzeMatch, GenerateReport
│   │   └── dto/         # Data transfer objects
│   ├── infrastructure/  # Technical implementation
│   │   ├── ml/          # YOLO detector, tracker
│   │   ├── video/       # Video processing
│   │   └── monitoring/  # Performance metrics
│   └── interfaces/      # External interfaces
│       ├── api/         # REST API
│       └── cli/         # Command line interface
├── data/
│   └── samples/         # Sample videos
├── models/
│   └── weights/         # Pre-trained models
├── outputs/             # Analysis results
├── tests/               # Test suite
└── docs/               # Documentation
```

## 🔧 Advanced Features

### Player Tracking with Statistics
```python
python player_stats_fixed.py
```
- Individual player tracking with unique IDs
- Speed calculation (m/s)
- Distance covered
- Sprint detection
- Movement trails visualization

### Tactical Analysis
```python
python tactical_view.py
```
- Real-time 2D minimap
- Team positioning
- Formation analysis
- Ball position tracking

### Heatmap Generation
```python
python formation_fixed.py
```
- Player density visualization
- Movement patterns
- Zone occupation analysis

### Event Detection
```python
python event_detection.py
```
- Ball out of bounds detection
- Goal opportunity recognition
- Pass detection
- Possession changes

## 🏗️ Architecture

The system follows **Clean Architecture** principles with **Domain-Driven Design**:

```
┌─────────────────────────────────────┐
│         Interface Layer             │  ← API, CLI, Web UI
├─────────────────────────────────────┤
│        Application Layer            │  ← Use Cases, DTOs
├─────────────────────────────────────┤
│          Domain Layer               │  ← Entities, Business Logic
├─────────────────────────────────────┤
│      Infrastructure Layer           │  ← ML Models, Video, Storage
└─────────────────────────────────────┘
```

### Design Patterns Used
- **Repository Pattern** - Data access abstraction
- **Strategy Pattern** - Swappable detection algorithms
- **Observer Pattern** - Event-driven analytics
- **Factory Pattern** - Model creation

## 📈 Analytics Capabilities

### Performance Metrics
- Total distance covered
- Average/maximum speed
- Activity intensity zones:
  - Walking (<2 m/s)
  - Jogging (2-5 m/s)
  - Running (5-7 m/s)
  - Sprinting (>7 m/s)

### Tactical Insights
- Team formation detection
- Player position heatmaps
- Pass network analysis
- Ball possession statistics
- Zone occupation metrics

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest --cov=src --cov-report=html

# Benchmarks
python benchmark.py
```

## 🔬 Technical Details

### Algorithms Implemented
- **YOLOv8** - State-of-the-art object detection
- **ByteTrack-inspired** - Simplified multi-object tracking
- **IoU Matching** - Detection-track association
- **K-means Clustering** - Team classification
- **Kalman Filtering** - Trajectory smoothing (planned)

### Known Limitations
1. **Distance/Speed Accuracy** - Without camera calibration, metrics are approximate
2. **Player Re-identification** - Simple IoU matching may create duplicate IDs
3. **Occlusion Handling** - Players may lose tracking when overlapping
4. **Perspective Distortion** - Metrics vary based on field position

### Planned Improvements
- [ ] Camera calibration for accurate measurements
- [ ] DeepSORT integration for robust tracking
- [ ] Pose estimation for action recognition
- [ ] Multiple camera support
- [ ] Real-time streaming support
- [ ] Cloud deployment ready

## 🚢 Deployment

### Docker
```bash
docker build -t sports-analytics .
docker run -v $(pwd)/data:/app/data sports-analytics
```

### Edge Deployment (NVIDIA Jetson)
```bash
# Optimized for edge devices
python scripts/optimize_for_edge.py
```

## 📊 API Documentation

The system includes a REST API built with FastAPI:

### Endpoints
- `POST /analyze` - Submit video for analysis
- `GET /status/{job_id}` - Check processing status
- `GET /results/{job_id}` - Download results
- `WS /stream` - Real-time streaming analysis

### Example
```python
import requests

# Upload video
with open("match.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": f}
    )
    job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{job_id}")
print(status.json())
```

## 📚 Documentation

Detailed documentation available in the `docs/` directory:
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Algorithm Details](docs/algorithms.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) community
- Sports analytics research community

## 📧 Contact

For questions or collaboration: your.email@example.com

---
*Built with ❤️ for sports analytics enthusiasts*