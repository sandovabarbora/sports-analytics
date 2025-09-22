from ultralytics import YOLO
import time
import cv2

models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
video = cv2.VideoCapture('data/samples/soccer/synthetic_hd.mp4')

for model_name in models:
    model = YOLO(model_name)
    
    times = []
    for _ in range(30):
        ret, frame = video.read()
        if not ret:
            break
        start = time.perf_counter()
        model(frame, verbose=False)
        times.append(time.perf_counter() - start)
    
    fps = 1.0 / (sum(times) / len(times))
    print(f"{model_name}: {fps:.1f} FPS")
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

video.release()
