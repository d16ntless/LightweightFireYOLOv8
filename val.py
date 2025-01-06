import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/fire/rmb/wiou/WIoUv3 a=1.9 d=3/weights/best.pt')
    model.val(data='/study/datasets/fire/fire.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/fire',
              name='yolov8s-C2f-iRMB-Cascaded-BiLevelRoutingAttention-WIoUv3 a=1.9 d=3',
              )