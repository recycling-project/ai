from ultralytics import YOLO

def train_model(): 
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='C:/Program Files (x86)/__DuFrump__/DuFrump_산대특_과정/07_Final_Projects/Project02/recycling/ai/data/data.yaml',
        epochs=150,
        imgsz=416,
        batch=16,
        patience=30,
        workers=4, 
        project='eco_scan',
        name='eco_scan_weight',
        device='cpu',
        verbose=True, 
        exist_ok=True,
        iou=0.5
    )

    return results