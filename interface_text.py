import os
import glob
from ultralytics import YOLO

MODEL_WEIGHTS_PATH = 'C:/Program Files (x86)/__DuFrump__/DuFrump_산대특_과정/07_Final_Projects/Project02/recycling/ai/models/train_yolov8m/weights/best.pt'

TEST_DIR_PATH = 'C:/Program Files (x86)/__DuFrump__/DuFrump_산대특_과정/07_Final_Projects/Project02/recycling/ai/test_images/'

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
# ----------------------------------------------------------------------

if not os.path.exists(MODEL_WEIGHTS_PATH):
    print("❌ 오류: 모델 파일 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요!")
    exit()
if not os.path.exists(TEST_DIR_PATH):
    print("❌ 오류: 테스트 이미지 폴더를 찾을 수 없습니다. 경로를 다시 확인해주세요!")
    exit()

try:
    model = YOLO(MODEL_WEIGHTS_PATH)
    
    all_image_files = []
    for ext in IMAGE_EXTENSIONS:
        all_image_files.extend(glob.glob(os.path.join(TEST_DIR_PATH, ext)))
        
    if not all_image_files:
        print(f"❌ 오류: {TEST_DIR_PATH} 폴더에서 이미지 파일(.jpg, .png 등)을 찾을 수 없습니다.")
        exit()

    print(f"✅ 총 {len(all_image_files)}개 이미지에 대한 예측을 시작합니다.")
    print("=" * 60)

    for image_path in all_image_files:
        
        results = model.predict(
            source=image_path,
            # conf=0.25,
            # iou=0.7,
            save=True,
            exist_ok=True,
            name='batch_test_results'
        )
        
        print(f"\n--- 결과: {os.path.basename(image_path)} ---")
        
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0:
                print("➡️ 탐지된 객체가 없습니다.")
                continue
                
            sorted_indices = boxes.conf.argsort(descending=True)

            
            print(f"➡️ 총 {len(boxes)}개의 객체 탐지됨. (Top 3 순위 출력)")
            
            for i in range(min(3, len(boxes))):
                idx = sorted_indices[i]
                
                cls = int(boxes.cls[idx].item())       
                conf = float(boxes.conf[idx].item())   
                
                class_names = model.names
                
                print(f" - [순위 {i+1}] 객체: {class_names.get(cls, 'Unknown Class')}, 확신도: {conf:.4f}")
            
        print("-" * 60)
        
    print("\n💡 모든 추론 완료! 결과 이미지는 'runs/detect/batch_test_results' 폴더에 저장되었습니다.")

except Exception as e:
    print(f"\n❌ 추론 중 치명적인 오류 발생: {e}")