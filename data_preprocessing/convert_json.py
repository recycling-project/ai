import json
import os
import glob
import shutil
from PIL import Image
from tqdm import tqdm

src_root_dir = "C:/Users/admin/Desktop/dataset/val"  

dst_images_dir = "C:/Users/admin/Desktop/dataset/val/images_final"
dst_labels_dir = "C:/Users/admin/Desktop/dataset/val/labels_final"

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
    "화분": 9
}

os.makedirs(dst_images_dir, exist_ok=True)
os.makedirs(dst_labels_dir, exist_ok=True)

print("🔍 1단계: 흩어진 이미지 파일들의 주소를 파악합니다...")

image_path_map = {}
valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

for ext in valid_exts:
    found_imgs = glob.glob(os.path.join(src_root_dir, "**", ext), recursive=True)
    
    for img_path in found_imgs:
        if "images_yolo_v3" in img_path:
            continue
        
        file_name = os.path.basename(img_path)
        image_path_map[file_name] = img_path

print(f"👉 총 {len(image_path_map)}개의 이미지 주소를 확보했습니다.")


print("\n🔍 2단계: JSON 변환 및 데이터셋 구축 시작...")

json_files = glob.glob(os.path.join(src_root_dir, "**", "*.json"), recursive=True)

success_cnt = 0
fail_cnt = 0

for json_file in tqdm(json_files):
    if "labels_yolo_v3" in json_file:
        continue

    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            fail_cnt += 1
            continue

    file_name = data.get("FILE NAME") or data.get("FILE_NAME") or data.get("filename")
    if not file_name:
        file_name = os.path.splitext(os.path.basename(json_file))[0] + ".jpg"

    real_img_path = image_path_map.get(file_name)
    
    if not real_img_path:
        if file_name.lower().endswith('.jpg'):
             real_img_path = image_path_map.get(file_name.replace('.jpg', '.JPG'))
        elif file_name.endswith('.JPG'):
             real_img_path = image_path_map.get(file_name.replace('.JPG', '.jpg'))

    if not real_img_path:
        fail_cnt += 1
        continue

    try:
        with Image.open(real_img_path) as img:
            img_w, img_h = img.size
        
        dst_img_path = os.path.join(dst_images_dir, file_name)
        if not os.path.exists(dst_img_path):
            shutil.copy2(real_img_path, dst_img_path)
            
    except:
        fail_cnt += 1
        continue

    txt_content = []
    bounding_list = data.get("Bounding") or []
    
    for obj in bounding_list:
        class_name = obj.get("DETAILS")
        if class_name not in class_map:
            continue 
            
        class_id = class_map[class_name]
        
        x1, y1, x2, y2 = 0, 0, 0, 0
        
        poly_points = obj.get("PolygonPoint")
        if poly_points:
            xs, ys = [], []
            for pt_dict in poly_points:
                for v in pt_dict.values():
                    try:
                        px, py = map(int, v.split(','))
                        xs.append(px)
                        ys.append(py)
                    except: pass
            
            if xs and ys:
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

        if x1 == 0 and x2 == 0:
            try:
                x1 = int(obj.get('x1', 0))
                y1 = int(obj.get('y1', 0))
                x2 = int(obj.get('x2', 0))
                y2 = int(obj.get('y2', 0))
            except:
                continue

        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1
        box_x = x1 + (box_w / 2.0)
        box_y = y1 + (box_h / 2.0)
        
        norm_x = box_x / img_w
        norm_y = box_y / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h
        
        txt_content.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    if txt_content:
        save_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(dst_labels_dir, save_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))
        success_cnt += 1
    else:
        fail_cnt += 1

print("="*50)
print(f"✅ 모든 변환 종료!")
print(f"   - 성공 (YOLO 데이터셋 생성): {success_cnt}건")
print(f"   - 실패/건너뜀 (이미지 없음/클래스 없음): {fail_cnt}건")
print(f"📂 최종 이미지 경로: {dst_images_dir}")
print(f"📂 최종 라벨 경로 : {dst_labels_dir}")