import cv2


class Camera:
    def __init__(self, cam_preset_num=5):
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):
        index = 0
        arr = []
        while index < 10:  # 最多检测 10 个摄像头
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # 使用 Media Foundation 后端
            if not cap.isOpened():
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return len(arr), arr


if __name__ == '__main__':
    cam = Camera()
    cam_num, devices = cam.get_cam_num()
    print(cam_num, devices)
