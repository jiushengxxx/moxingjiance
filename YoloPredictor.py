from ultralytics import YOLO
import cv2
import torch

def setup_model(self, model_path):
    self.model = YOLO(model_path)  # 使用 YOLO 类加载模型
    if torch.cuda.is_available():
        self.model.to('cuda')  # 将模型移动到 GPU

def run(self):
    """运行检测任务"""
    if isinstance(self.source, int):  # 如果源是摄像头索引
        # 尝试使用 Media Foundation 后端
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            # 如果 Media Foundation 失败，尝试使用 OpenCV 默认后端
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.yolo2main_status_msg.emit('无法打开摄像头，请检查设备连接。')
                return

    while not self.stop_dtc:
        ret, frame = self.cap.read()
        if not ret:
            self.yolo2main_status_msg.emit('无法读取摄像头帧。')
            break

        # 将帧传递给 GUI 界面
        self.yolo2main_pre_img.emit(frame)  # 发送原始帧
        results = self.model(frame)  # 进行模型推理
        annotated_frame = results[0].plot()  # 获取带标注的帧
        self.yolo2main_res_img.emit(annotated_frame)  # 发送带标注的帧

    self.cap.release()
    self.yolo2main_status_msg.emit('摄像头已停止。')

    # ... 其他代码 ... 