import json
import os
import glob
from collections import Counter
from tqdm import tqdm

json_root_dir = "C:/Users/admin/Desktop/dataset/train/labels"
class_map = {
    "밥상": 0,
    "서랍장": 1,
    "소파": 2,
    "의자": 3,
    "장롱": 4,
    "책상": 5,
    "화장대": 6,
    "침대": 7,
    "두발자전거": 8, 
    "항아리": 9,
    "화분": 9,
}

print("버려진 데이터의 정체를 파악합니다...")
json_files = glob.glob(os.path.join(json_root_dir, "**", "*.json"), recursive=True)

skipped_classes = Counter()
total_skipped_files = 0

for json_file in tqdm(json_files):
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            continue

    bounding_list = data.get("Bounding") or []
    
    has_valid_object = False
    file_skipped_objects = []

    for obj in bounding_list:
        class_name = obj.get("DETAILS")
        if class_name in class_map:
            has_valid_object = True
        else:
            if class_name:
                file_skipped_objects.append(class_name)
    
    if not has_valid_object and len(bounding_list) > 0:
        total_skipped_files += 1
        for bad_class in file_skipped_objects:
            skipped_classes[bad_class] += 1

print("\n" + "="*40)
print(f"TXT가 생성되지 않은 파일 수: 약 {total_skipped_files}개")
print("이 품목들이 `class_map`에 없어서 버려졌습니다! (상위 20개)")
print("-" * 40)
for name, count in skipped_classes.most_common(20):
    print(f"{name}: {count}회 등장")
print("="*40)