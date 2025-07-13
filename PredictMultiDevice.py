import torch
import cv2
import os
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator
from utils.torch_utils import select_device

# 교차 여부 판단 함수 (조금이라도 겹치면 True)
def box_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

# 경로 설정
input_folder = "test_image/input"
output_folder = "test_image/output"
os.makedirs(output_folder, exist_ok=True)

# 디바이스 선택
device = select_device('cpu')

# 1차 모델 로드 (device 판단용)
model_device = DetectMultiBackend("runs/train/connect_yolov5s_results/weights/best_device.pt", device=device)
device_names = model_device.names

# 클래스별 2차 모델 경로
sub_model_paths = {
    'BK20S-T2': 'runs/train/connect_yolov5s_results/weights/best_BK20_v2.pt',
    'BS32c-6A': 'runs/train/connect_yolov5s_results/weights/best_BS32c_v2.pt',
    'WAGO750-1505': 'runs/train/connect_yolov5s_results/weights/best_wago_v2.pt',
}

# 이미지 확장자
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# 이미지 반복
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(valid_exts):
        continue

    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    orig_img = cv2.imread(input_path)
    if orig_img is None:
        print(f"이미지 로드 실패: {input_path}")
        continue

    # 전처리
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0

    # 1차 추론
    with torch.no_grad():
        pred1 = model_device(img_tensor, augment=False, visualize=False)
        preds1 = non_max_suppression(pred1, conf_thres=0.5, iou_thres=0.45)
        pred1 = preds1[0] if len(preds1) > 0 else None

    annotator = Annotator(orig_img.copy(), line_width=2, example=str(device_names))
    sub_models_loaded = {}

    if pred1 is not None and len(pred1):
        pred1[:, :4] = scale_boxes(img_tensor.shape[2:], pred1[:, :4], orig_img.shape[:2]).round()
        for *xyxy, conf, cls in pred1:
            cls_name = device_names[int(cls)]
            label = f'{cls_name} {conf:.2f}'

            # 1차 클래스 색상
            if cls_name == 'WAGO750-1505':
                color = (255, 0, 0)   # 파랑
            elif cls_name == 'BK20S-T2':
                color = (0, 255, 255) # 노랑
            elif cls_name == 'BS32c-6A':
                color = (0, 165, 255) # 주황
            else:
                color = (255, 255, 255)

            annotator.box_label(xyxy, label, color=color)

            # 2차 모델 경로가 있으면 2차 추론 수행
            if cls_name in sub_model_paths:
                weight_path = sub_model_paths[cls_name]
                if cls_name not in sub_models_loaded:
                    model_sub = DetectMultiBackend(weight_path, device=device)
                    sub_models_loaded[cls_name] = (model_sub, model_sub.names)
                else:
                    model_sub, sub_names = sub_models_loaded[cls_name]

                # 2차 추론
                with torch.no_grad():
                    pred2 = model_sub(img_tensor, augment=False, visualize=False)
                    preds2 = non_max_suppression(pred2, conf_thres=0.5, iou_thres=0.45)
                    pred2 = preds2[0] if len(preds2) > 0 else None

                # 2차 결과 시각화 (겹치는 경우에만)
                if pred2 is not None and len(pred2):
                    pred2[:, :4] = scale_boxes(img_tensor.shape[2:], pred2[:, :4], orig_img.shape[:2]).round()
                    for *xyxy2, conf2, cls2 in pred2:
                        if not box_overlap(xyxy2, xyxy):  # 겹치는 경우만
                            continue
                        cls_name2 = sub_models_loaded[cls_name][1][int(cls2)]
                        label2 = f'{cls_name2} {conf2:.2f}'
                        if cls_name2 == 'connect':
                            color2 = (0, 255, 0)  # 초록
                        elif cls_name2 == 'disconnect':
                            color2 = (0, 0, 255)  # 빨강
                        else:
                            color2 = (200, 200, 200)
                        annotator.box_label(xyxy2, label2, color=color2)

    # 결과 저장
    result_img = annotator.result()
    cv2.imwrite(output_path, result_img)
    print(f"저장 완료: {output_path}")