from ultralytics import YOLO

def train_model(): 
    model = YOLO('yolov8s-seg.pt')

    results = model.train(
        data='./data/data.yaml',  # 데이터셋 경로
        epochs=150,
        imgsz=640,
        batch=32,
        project='eco_scan',
        name='eco_scan_weight',
        device=0,
        verbose=True, 
        exist_ok=True,
        iou=0.5
    )

    return results