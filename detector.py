from ultralytics import YOLO
import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        self.model = YOLO(model_path)
        
    def detect_image(self, image_path):
        results = self.model(image_path)
        return results[0].plot()
        
    def detect_stream(self, source):
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            yield results[0].plot()
        cap.release() 