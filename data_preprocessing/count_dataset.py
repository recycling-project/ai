import os
import glob
from collections import Counter
from tqdm import tqdm

label_dir = "C:/Users/admin/Desktop/dataset/train/labels_final"

id_to_name = {
    0: "밥상",
    1: "서랍장",
    2: "소파",
    3: "의자",
    4: "장롱",
    5: "책상",
    6: "화장대",
    7: "침대",
    8: "두발자전거",
    9: "항아리"
}

txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
print(f"📂 분석할 파일(이미지) 개수: {len(txt_files)}개")

class_counter = Counter()
total_objects = 0

print("🔍 데이터를 분석 중입니다...")

for txt_file in tqdm(txt_files):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_id = int(parts[0])
                class_counter[class_id] += 1
                total_objects += 1

print("\n" + "="*40)
print(f"📊 [데이터셋 분포 현황] (총 객체 수: {total_objects}개)")
print("="*40)

sorted_ids = sorted(class_counter.keys())

for cls_id in sorted_ids:
    count = class_counter[cls_id]
    name = id_to_name.get(cls_id, f"알 수 없음(ID:{cls_id})")
    
    ratio = (count / total_objects) * 100
    
    bar = "█" * int(ratio // 2) 
    
    print(f"{cls_id}번 [{name}]: {count}개 ({ratio:.1f}%) {bar}")

print("="*40)

print("\n📢 [진단 결과]")
min_count = 100
warning_classes = [id_to_name.get(k, k) for k, v in class_counter.items() if v < min_count]

if warning_classes:
    print(f"⚠️ 데이터가 너무 적은 품목({min_count}개 미만): {warning_classes}")
    print("   -> 데이터를 더 수집하거나, 데이터 증강(Augmentation)이 필요할 수 있습니다.")
else:
    print("✅ 모든 품목의 데이터 양이 양호합니다!")