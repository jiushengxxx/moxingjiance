import cv2


class Camera:
    def __init__(self, cam_preset_num=5):
        self.cam_preset_num = cam_preset_num
        self.cameras = {}  # 用于存储摄像头对象
        self.current_camera = None  # 当前使用的摄像头

    def get_cam_num(self):
        index = 0
        arr = []
        while index < 10:  # 最多检测 10 个摄像头
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)  # 使用 Media Foundation 后端
            if not cap.isOpened():
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return len(arr), arr

    def add_camera(self, camera_id):
        """添加摄像头"""
        if camera_id not in self.cameras:
            self.cameras[camera_id] = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
            if not self.cameras[camera_id].isOpened():
                raise Exception(f"无法打开摄像头 {camera_id}")

    def switch_camera(self, camera_id):
        """切换到指定摄像头"""
        if camera_id in self.cameras:
            self.current_camera = self.cameras[camera_id]
        else:
            raise Exception(f"摄像头 {camera_id} 未找到")


if __name__ == '__main__':
    cam = Camera()
    cam_num, devices = cam.get_cam_num()
    print(cam_num, devices)
