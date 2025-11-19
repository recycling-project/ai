import os
import cv2
import glob
import albumentations as A
from tqdm import tqdm

# =========================================================
# 1. 설정 (부족한 클래스를 여기서 정의하세요!)
# =========================================================
# 증강할 대상 클래스 번호와 목표 개수 비율 (예: 10배로 늘려라)
target_classes = {
    7: 15,  # 침대(80개) -> x15배 -> 약 1200개 확보 목표
    6: 5,   # 화장대(239개) -> x5배 -> 약 1200개 확보 목표
    4: 2,    # 장롱(638개) -> x2배 -> 약 1200개 확보 목표
    8: 2    # 두발자전거(937개) -> x2배 -> 약 1800개 확보 목표
}

# 데이터셋 경로 (final 폴더)
img_dir = "C:/Users/admin/Desktop/dataset/train/images_final"
txt_dir = "C:/Users/admin/Desktop/dataset/train/labels_final"

# =========================================================
# 2. 증강 파이프라인 정의 (Albumentations)
# =========================================================
# 너무 심하게 변형하면 오히려 학습을 방해하므로 적당하게 설정
transform = A.Compose([
    A.HorizontalFlip(p=0.5),       # 좌우 반전
    A.RandomBrightnessContrast(p=0.5), # 밝기/대비 조절
    A.Rotate(limit=15, p=0.5),     # 살짝 회전 (-15도 ~ 15도)
    A.GaussianBlur(p=0.3),         # 흐리게 (노이즈 효과)
    A.CLAHE(p=0.3),                # 선명하게
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# =========================================================
# 3. 증강 로직 시작
# =========================================================
txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
print(f"🔍 총 {len(txt_files)}개의 라벨 파일을 검사합니다...")

aug_cnt = 0

for txt_path in tqdm(txt_files):
    # 1) 라벨 파일 읽기
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 이 사진에 '타겟 클래스'가 있는지 확인
    has_target = False
    bboxes = []
    class_labels = []
    
    multiply_factor = 0 # 몇 배로 늘릴지 결정 (가장 희귀한 객체 기준)

    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        cls_id = int(parts[0])
        # 좌표 (x, y, w, h)
        bbox = [float(x) for x in parts[1:]]
        
        bboxes.append(bbox)
        class_labels.append(cls_id)
        
        # 만약 이 줄의 객체가 '증강 대상'이라면?
        if cls_id in target_classes:
            has_target = True
            # 여러 타겟이 같이 있을 경우, 더 많이 늘려야 하는 녀석을 기준으로 잡음
            multiply_factor = max(multiply_factor, target_classes[cls_id])

    # 타겟이 없는 평범한 사진이면 패스
    if not has_target:
        continue

    # 2) 이미지 읽기
    file_name = os.path.basename(txt_path).replace('.txt', '.jpg') # 확장자 주의
    img_path = os.path.join(img_dir, file_name)
    
    if not os.path.exists(img_path):
        continue
        
    image = cv2.imread(img_path)
    if image is None: continue
    
    # 3) 증강 생성 (multiply_factor 만큼 반복)
    for i in range(multiply_factor):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            
            # 만약 증강 과정에서 박스가 사라졌다면 저장 안 함
            if len(aug_bboxes) == 0: continue

            # 4) 파일 저장 (이름 뒤에 _aug_0, _aug_1 붙임)
            name_base = os.path.splitext(file_name)[0]
            new_name = f"{name_base}_aug_{i}"
            
            # 이미지 저장
            cv2.imwrite(os.path.join(img_dir, new_name + ".jpg"), aug_img)
            
            # 라벨 저장
            new_txt_content = []
            for cls, bbox in zip(class_labels, aug_bboxes):
                # YOLO 형식 유지 (cls x y w h)
                # albumentations가 가끔 범위를 살짝 넘길 때가 있어 클리핑(0~1)
                x, y, w, h = bbox
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)
                new_txt_content.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
            with open(os.path.join(txt_dir, new_name + ".txt"), 'w') as f:
                f.write('\n'.join(new_txt_content))
            
            aug_cnt += 1
            
        except Exception as e:
            print(f"Error augmenting {file_name}: {e}")

print("="*40)
print(f"✅ 증강 완료! 총 {aug_cnt}개의 새로운 데이터가 생성되었습니다.")
print(f"📂 저장 위치: {img_dir}")