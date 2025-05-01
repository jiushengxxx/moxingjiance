import cv2
import time
from PySide6.QtCore import QThread, Signal, QObject


class Camera(QThread):
    # 信号：用于传递视频帧和摄像头状态
    frame_ready = Signal(object)  # 视频帧信号
    camera_status = Signal(str)   # 摄像头状态信号

    def __init__(self, device_id=0, frame_width=640, frame_height=480):
        super().__init__()
        self.device_id = device_id  # 摄像头设备ID
        self.frame_width = frame_width  # 视频帧宽度
        self.frame_height = frame_height  # 视频帧高度
        self.cap = None  # 摄像头对象
        self.running = False  # 摄像头运行状态

    def run(self):
        """启动摄像头并读取视频帧"""
        self.running = True
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.camera_status.emit(f"无法打开摄像头 {self.device_id}")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera_status.emit(f"摄像头 {self.device_id} 已启动")

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.camera_status.emit(f"摄像头 {self.device_id} 断开连接")
                    break
                self.frame_ready.emit(frame)  # 发送视频帧信号
                time.sleep(0.03)  # 控制帧率

        except Exception as e:
            self.camera_status.emit(f"摄像头 {self.device_id} 打开失败: {str(e)}")
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.camera_status.emit(f"摄像头 {self.device_id} 已关闭")

    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def set_device(self, device_id):
        """切换摄像头设备"""
        self.device_id = device_id
        if self.running:
            self.stop()
            self.start()


class CameraManager(QObject):
    """摄像头管理类，支持多摄像头切换"""
    def __init__(self):
        super().__init__()
        self.cameras = {}  # 摄像头字典，key为设备ID，value为Camera对象
        self.current_camera = None  # 当前使用的摄像头

    def add_camera(self, device_id):
        """添加摄像头"""
        if device_id not in self.cameras:
            camera = Camera(device_id)
            self.cameras[device_id] = camera
            return camera
        return None

    def remove_camera(self, device_id):
        """移除摄像头"""
        if device_id in self.cameras:
            self.cameras[device_id].stop()
            del self.cameras[device_id]

    def switch_camera(self, device_id):
        """切换摄像头"""
        if device_id in self.cameras:
            if self.current_camera:
                self.current_camera.stop()
            self.current_camera = self.cameras[device_id]
            self.current_camera.start()

    def get_available_cameras(self):
        """获取可用的摄像头设备"""
        available_cameras = []
        for device_id in range(5):  # 假设最多检测5个摄像头
            try:
                cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    available_cameras.append(device_id)
                    cap.release()
            except Exception as e:
                print(f"检测摄像头 {device_id} 失败: {str(e)}")
        return available_cameras

    def start_all_cameras(self):
        """启动所有摄像头"""
        for camera in self.cameras.values():
            camera.start()

    def stop_all_cameras(self):
        """停止所有摄像头"""
        for camera in self.cameras.values():
            camera.stop()


if __name__ == '__main__':
    cam = Camera()
    cam_num, devices = cam.get_cam_num()
    print(cam_num, devices)
