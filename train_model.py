from ultralytics import YOLO
import argparse
import torch
import os


def train_model(yaml_file, epochs=100, imgsz=640, save_dir='./models'):
    """
    使用 YOLOv8 训练模型
    :param yaml_file: YAML 配置文件路径
    :param epochs: 训练轮数，默认为 100
    :param imgsz: 图像大小，默认为 640
    :param save_dir: 模型保存路径，默认为 './models'
    """
    try:
        # 检查是否有可用的 GPU
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device = 'cuda:0'  # 使用第一个 GPU
            else:
                device = 'cpu'
        else:
            device = 'cpu'
        print(f"使用设备: {device}")

        # 加载 YOLOv8 模型
        model = YOLO("yolov8n.pt")  # 使用预训练的 YOLOv8n 模型
        print(f"开始训练模型，配置文件: {yaml_file}")

        # 启动训练，指定使用 GPU
        results = model.train(data=yaml_file, epochs=epochs, imgsz=imgsz, device=device)
        print("训练完成！")

        # 保存训练好的模型
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, 'trained_model.pt')
        model.export(format='pt', path=model_path)
        print(f"模型已保存到: {model_path}")

        # 输出训练结果
        print(f"训练结果: {results}")

    except Exception as e:
        print(f"训练失败: {str(e)}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLOv8 模型训练脚本")
    parser.add_argument("--yaml", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认为 100")
    parser.add_argument("--imgsz", type=int, default=640, help="图像大小，默认为 640")
    parser.add_argument("--save_dir", type=str, default='./models', help="模型保存路径，默认为 './models'")
    args = parser.parse_args()

    # 启动训练
    train_model(args.yaml, args.epochs, args.imgsz, args.save_dir) 