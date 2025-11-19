import os
import glob
import shutil
import random
from tqdm import tqdm
from collections import Counter

# =========================================================
# 1. 설정 (경로 확인 필수!)
# =========================================================
# 원본 데이터셋 경로 (v3 폴더)
src_img_dir = "C:/Users/admin/Desktop/dataset/train/images_final"
src_lbl_dir = "C:/Users/admin/Desktop/dataset/train/labels_final"

# 균형 맞춘 데이터를 저장할 새로운 폴더
dst_img_dir = "C:/Users/admin/Desktop/dataset/train/images_balanced"
dst_lbl_dir = "C:/Users/admin/Desktop/dataset/train/labels_balanced"

# [목표] 각 클래스당 최대 허용 개수
LIMIT_PER_CLASS = 1100 

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# =========================================================
# 2. 파일 선별 로직 (Under-sampling)
# =========================================================
print("🔍 데이터를 스캔하고 섞는 중입니다...")

# 라벨 파일 목록 가져오기
txt_files = glob.glob(os.path.join(src_lbl_dir, "*.txt"))
random.shuffle(txt_files) # 랜덤하게 섞어야 공평하게 뽑힘!

# 현재까지 담은 개수를 셀 카운터
current_counts = Counter()
selected_files = []

print("⚖️ 데이터 선별 시작 (목표: 클래스당 1,100개 이하)...")

for txt_path in tqdm(txt_files):
    # 1) 파일 안에 어떤 물건이 있는지 확인
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # 이 파일에 포함된 클래스들 (예: [0, 0, 3] -> {0, 3})
    classes_in_file = set()
    for line in lines:
        parts = line.strip().split()
        if parts:
            classes_in_file.add(int(parts[0]))
    
    # 2) "이 파일을 가져갈까 말까?" 결정
    # 조건: 이 파일에 있는 물건 중, 아직 1100개가 안 찬 게 '하나라도' 있으면 가져간다.
    should_keep = False
    for cls_id in classes_in_file:
        if current_counts[cls_id] < LIMIT_PER_CLASS:
            should_keep = True
            break # 하나라도 부족하면 즉시 채택!
    
    # 3) 가져가기로 결정했다면?
    if should_keep:
        selected_files.append(txt_path)
        # 카운터 업데이트 (이 파일에 들어있는 모든 물건 개수 추가)
        for cls_id in classes_in_file:
            current_counts[cls_id] += 1

# =========================================================
# 3. 파일 복사 (이사하기)
# =========================================================
print(f"\n🚚 선별된 {len(selected_files)}개의 파일을 복사합니다...")

for txt_path in tqdm(selected_files):
    # 라벨 복사
    file_name = os.path.basename(txt_path)
    shutil.copy2(txt_path, os.path.join(dst_lbl_dir, file_name))
    
    # 이미지 복사 (확장자 찾기)
    img_name_no_ext = os.path.splitext(file_name)[0]
    
    # 원본 이미지 폴더에서 같은 이름의 이미지 찾기
    # (jpg, png 등 확장자가 다를 수 있으니 glob으로 찾음)
    found_imgs = glob.glob(os.path.join(src_img_dir, img_name_no_ext + ".*"))
    
    if found_imgs:
        shutil.copy2(found_imgs[0], os.path.join(dst_img_dir, os.path.basename(found_imgs[0])))

# =========================================================
# 4. 최종 결과 리포트
# =========================================================
print("\n" + "="*40)
print("📊 [최종 균형 데이터셋 분포]")
print("="*40)
for cls_id in sorted(current_counts.keys()):
    print(f"CLASS {cls_id}: {current_counts[cls_id]}개")
print("="*40)
print(f"📂 저장 위치: {dst_img_dir}")