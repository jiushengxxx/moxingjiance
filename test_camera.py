import cv2

# 尝试打开默认摄像头（索引为 0）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头，请检查设备连接。")
else:
    print("摄像头已成功打开。")
    ret, frame = cap.read()
    if ret:
        print("摄像头帧读取成功。")
    else:
        print("无法读取摄像头帧。")
    cap.release()