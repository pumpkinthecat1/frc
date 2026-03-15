# detect_balls.py
from ultralytics import YOLO

class BallDetector:
    def __init__(self, model_path="runs/train/frc2026_balls/weights/best.pt"):
        """
        Initialize the ball detector with a YOLOv8 model.
        Replace model_path with your trained model path.
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect balls in a single frame.
        Returns the number of balls detected.
        """
        results = self.model(frame)
        count = 0
        for r in results:
            boxes = r.boxes
            count += len(boxes)
        return count
