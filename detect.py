import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/foggia/yolov8n/weights/best.pt') # select your model.pt path
    model.predict(source='D:/study/datasets/foggia/images/valid',
                  imgsz=640,
                  project='runs/detect/foggia',
                  name='yolov8n',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # visualize=True # visualize model features maps
                )