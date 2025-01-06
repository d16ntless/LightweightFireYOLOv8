import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/study/datasets/foggia/foggia.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume='runs/train/foggia/aaa/yolov8s-ircb-BiLevelRoutingAttention WIoUv3 a=2.0 d=3.0 epoch300/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train/foggia',
                name='yolov8n',
                )