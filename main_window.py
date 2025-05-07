from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                              QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QComboBox, QWidget, QStyle)
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon
from PySide6.QtCore import QThread, Signal, Qt
import cv2
import numpy as np
from detector import ObjectDetector
from config import MODEL_OPTIONS, DEFAULT_MODEL
import os

class DetectionThread(QThread):
    finished = Signal(np.ndarray)
    
    def __init__(self, detector, source):
        super().__init__()
        self.detector = detector
        self.source = source
        self.running = True
        
    def run(self):
        for frame in self.detector.detect_stream(self.source):
            if not self.running:
                break
            self.finished.emit(frame)
            
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Detector")
        self.setGeometry(100, 100, 1000, 600)
        self.detector = ObjectDetector(DEFAULT_MODEL)
        if not os.path.exists("icons"):
            os.makedirs("icons")
        self.init_ui()
        
    def init_ui(self):
        # 设置全局字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)
        
        # 主显示区域 - 添加边框和背景
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #d0d0d0;
                border-radius: 5px;
                color: #888;
                font-size: 16px;
            }
        """)
        
        # 控制按钮 - 添加图标和样式
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_OPTIONS)
        self.model_combo.setStyleSheet("""
            QComboBox {
                padding: 5px 10px 5px 30px;
                min-width: 150px;
                background-position: left center;
                background-repeat: no-repeat;
                padding-left: 30px;
            }
        """)
        
        # 创建带图标的按钮
        self.btn_image = QPushButton(
            self.style().standardIcon(QStyle.SP_FileIcon), 
            "检测图片"
        )
        self.btn_video = QPushButton(
            self.style().standardIcon(QStyle.SP_MediaPlay), 
            "检测视频"
        )
        self.btn_camera = QPushButton(
            self.style().standardIcon(QStyle.SP_DriveCDIcon),  # 使用光盘图标替代摄像头
            "摄像头检测"
        )
        self.btn_stop = QPushButton(
            self.style().standardIcon(QStyle.SP_MediaStop), 
            "停止"
        )
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                padding: 8px 15px;
                border-radius: 4px;
                background-color: #4CAF50;
                color: white;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        for btn in [self.btn_image, self.btn_video, self.btn_camera]:
            btn.setStyleSheet(button_style)
        
        # 停止按钮特殊样式
        self.btn_stop.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                border-radius: 4px;
                background-color: #f44336;
                color: white;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        
        # 布局 - 添加间距和边距
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(self.btn_image)
        control_layout.addWidget(self.btn_video)
        control_layout.addWidget(self.btn_camera)
        control_layout.addWidget(self.btn_stop)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.display_label, 1)  # 添加伸缩因子
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # 设置窗口最小大小
        self.setMinimumSize(800, 600)
        
        # 连接信号
        self.btn_image.clicked.connect(self.detect_image)
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_camera.clicked.connect(self.detect_camera)
        self.btn_stop.clicked.connect(self.stop_detection)
        
        # 设置窗口图标
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")
        
        # 显示区域添加提示文本
        self.display_label.setText("请选择检测源")
        self.display_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #d0d0d0;
                border-radius: 5px;
                color: #888;
                font-size: 16px;
            }
        """)
        
    def change_model(self, model_name):
        self.detector = ObjectDetector(model_name)
        
    def detect_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg)")
        if file_path:
            self.stop_detection()
            frame = self.detector.detect_image(file_path)
            self.display_results(frame)
            
    def detect_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            self.start_detection(file_path)
            
    def detect_camera(self):
        self.start_detection(0)  # 0表示默认摄像头
        
    def start_detection(self, source):
        self.stop_detection()
        self.detection_thread = DetectionThread(self.detector, source)
        self.detection_thread.finished.connect(self.display_results)
        self.detection_thread.start()
        
    def stop_detection(self):
        if hasattr(self, 'detection_thread') and self.detection_thread.isRunning():
            self.detection_thread.stop()
        self.statusBar().showMessage("已停止")
        
    def display_results(self, frame):
        # 转换OpenCV图像为Qt图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.display_label.width(), 
            self.display_label.height(),
            Qt.KeepAspectRatio))
        self.statusBar().showMessage("检测中...")
            
    def closeEvent(self, event):
        self.stop_detection()
        event.accept() 
    # ... 保留原有UI代码，但修改相关方法使用新的detector类 ... 