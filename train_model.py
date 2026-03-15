# train_model.py
from ultralytics import YOLO
import os

def train():
    # 1. Load the base YOLOv8 model (the "brain" we will train)
    model = YOLO("yolov8n.pt") 

    # 2. Train the model
    # data="dataset/data.yaml" tells the AI where your ball images are
    # epochs=50 means it will look at the images 50 times to learn
    results = model.train(
        data="dataset/data.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        name="frc2026_balls"
    )

    print("Training complete! Your new model is saved in: runs/detect/frc2026_balls/weights/best.pt")

if __name__ == "__main__":
    train()
