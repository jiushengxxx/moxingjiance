import sys
import cv2
import torch
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal


class YoloDetectionThread(QThread):
    raw_frame_signal = Signal(QImage)
    detected_frame_signal = Signal(QImage)

    def __init__(self, model_path, camera_index=0):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.is_running = True
        self.model = None

    def setup_model(self):
        self.model = YOLO(self.model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')

    def run(self):
        self.setup_model()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return

        while self.is_running:
            ret, frame = cap.read()
            if ret:
                # 发送原始帧
                raw_qimage = self.convert_to_qimage(frame)
                self.raw_frame_signal.emit(raw_qimage)

                # 进行目标检测
                results = self.model(frame)
                annotated_frame = results[0].plot()
                detected_qimage = self.convert_to_qimage(annotated_frame)
                self.detected_frame_signal.emit(detected_qimage)
        cap.release()

    def convert_to_qimage(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def stop(self):
        self.is_running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.init_yolo_thread()

    def initUI(self):
        layout = QHBoxLayout()
        self.raw_frame_label = QLabel()
        self.detected_frame_label = QLabel()
        layout.addWidget(self.raw_frame_label)
        layout.addWidget(self.detected_frame_label)
        self.setLayout(layout)

    def init_yolo_thread(self):
        self.yolo_thread = YoloDetectionThread(model_path='yolov8n.pt')
        self.yolo_thread.raw_frame_signal.connect(self.show_raw_frame)
        self.yolo_thread.detected_frame_signal.connect(self.show_detected_frame)
        self.yolo_thread.start()

    def show_raw_frame(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.raw_frame_label.setPixmap(pixmap)
        self.raw_frame_label.setScaledContents(True)

    def show_detected_frame(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.detected_frame_label.setPixmap(pixmap)
        self.detected_frame_label.setScaledContents(True)

    def closeEvent(self, event):
        self.yolo_thread.stop()
        self.yolo_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    