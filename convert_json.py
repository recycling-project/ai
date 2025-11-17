import json
import os
import glob
import shutil  # 파일 복사를 위한 도구
from PIL import Image
from tqdm import tqdm  # 진행률 표시바 (pip install tqdm)

# -------------------------------------------------------
# 1. 경로 설정 (사용자 환경에 맞게 수정 필수!)
# -------------------------------------------------------
# [입력] 원본 데이터가 흩어져 있는 최상위 폴더
src_root_dir = "C:/Users/admin/Desktop/dataset/train"  

# [출력] YOLO용으로 깔끔하게 모을 폴더 (자동 생성됨)
dst_images_dir = "C:/Users/admin/Desktop/dataset/train/images_yolo"
dst_labels_dir = "C:/Users/admin/Desktop/dataset/train/labels_yolo"

# [중요] data.yaml 번호와 일치
class_map = {
    "밥상": 0,
    "서랍장": 1,
    "소파": 2,
    "의자": 3,
    "장롱": 4,
    "책상": 5,
    "화장대": 6,
    "침대": 7,
    "자전거": 8, 
    "항아리": 9,
}

# 폴더 생성
os.makedirs(dst_images_dir, exist_ok=True)
os.makedirs(dst_labels_dir, exist_ok=True)

# -------------------------------------------------------
# 2. 이미지 지도(Map) 만들기 (속도 향상 핵심)
# -------------------------------------------------------
print("🔍 1단계: 흩어진 이미지 파일들의 위치를 파악합니다...")

# 이미지 확장자들
img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_path_map = {} # { '파일명.jpg': '실제/경로/파일명.jpg' }

for ext in img_extensions:
    # src_root_dir 아래 모든 폴더를 뒤져서 이미지를 찾음
    found_imgs = glob.glob(os.path.join(src_root_dir, "**", ext), recursive=True)
    
    for img_path in found_imgs:
        # 'images_yolo' 폴더(우리가 만들고 있는 폴더) 안에 있는 건 제외
        if "images_yolo" in img_path:
            continue
            
        file_name = os.path.basename(img_path)
        # 같은 이름의 파일이 있을 경우를 대비해 덮어쓰거나 로그 남김 (여기선 덮어씀)
        image_path_map[file_name] = img_path

print(f"👉 총 {len(image_path_map)}개의 이미지 위치를 등록했습니다.")


# -------------------------------------------------------
# 3. JSON 변환 및 파일 복사 시작
# -------------------------------------------------------
print("\n🔍 2단계: JSON을 찾아 변환하고, 짝꿍 이미지를 복사합니다...")

json_files = glob.glob(os.path.join(src_root_dir, "**", "*.json"), recursive=True)
print(f"👉 총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

success_cnt = 0
fail_cnt = 0

# tqdm으로 진행률 바 표시
for json_file in tqdm(json_files):
    # 이미 변환된 폴더에 있는 json은 패스
    if "labels_yolo" in json_file:
        continue

    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            fail_cnt += 1
            continue

    # 1) 파일 이름 확인
    file_name = data.get("FILE NAME")
    if not file_name:
        file_name = data.get("FILE_NAME") or data.get("filename")
    
    if not file_name:
        # 파일명 없으면 JSON 파일명으로 추측
        base = os.path.basename(json_file)
        file_name = os.path.splitext(base)[0] + ".jpg"

    # 2) 이미지 위치 찾기 (아까 만든 지도 이용)
    real_img_path = image_path_map.get(file_name)
    
    # 못 찾았으면 대소문자 바꿔서 한 번 더 시도 (.jpg <-> .JPG)
    if not real_img_path:
        if file_name.lower().endswith('.jpg'):
             real_img_path = image_path_map.get(file_name.replace('.jpg', '.JPG'))
        elif file_name.endswith('.JPG'):
             real_img_path = image_path_map.get(file_name.replace('.JPG', '.jpg'))

    if not real_img_path:
        # print(f"🚨 이미지 없음(Skip): {file_name}") # 너무 많이 뜨면 주석 처리
        fail_cnt += 1
        continue

    # 3) 이미지 크기 확인 및 복사
    try:
        with Image.open(real_img_path) as img:
            img_w, img_h = img.size
            
        # [핵심] 이미지를 YOLO 폴더로 복사 (이미 있으면 건너뜀)
        dst_img_path = os.path.join(dst_images_dir, file_name)
        if not os.path.exists(dst_img_path):
            shutil.copy2(real_img_path, dst_img_path)
            
    except Exception as e:
        # 이미지 깨짐 등
        fail_cnt += 1
        continue

    # 4) 좌표 변환
    txt_content = []
    bounding_list = data.get("Bounding") or []
    
    for obj in bounding_list:
        class_name = obj.get("DETAILS")
        if class_name not in class_map:
            continue 

        class_id = class_map[class_name]
        
        try:
            x1 = int(obj['x1'])
            y1 = int(obj['y1'])
            x2 = int(obj['x2'])
            y2 = int(obj['y2'])
        except:
            continue

        dw = 1. / img_w
        dh = 1. / img_h
        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        
        x_center *= dw
        w *= dw
        y_center *= dh
        h *= dh
        
        txt_content.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # 5) 라벨 저장
    if txt_content:
        save_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(dst_labels_dir, save_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))
        success_cnt += 1
    else:
        # 내용은 없지만 이미지는 복사된 경우 -> 이미지를 다시 지울지 선택 (여기선 둠)
        fail_cnt += 1

print("="*50)
print(f"✅ 정리 완료!")
print(f"   - 성공(세트 생성): {success_cnt}건")
print(f"   - 실패/건너뜀: {fail_cnt}건")
print(f"📂 모인 이미지: {dst_images_dir}")
print(f"📂 모인 라벨: {dst_labels_dir}")