import torch
import cv2
import numpy as np
from models.common     import DetectMultiBackend
from utils.general     import non_max_suppression, scale_boxes
from utils.plots       import Annotator
from utils.torch_utils import select_device
from camera import ZedCamera

from pathlib import Path
import pathlib

pathlib.WindowsPath = pathlib.PosixPath
usr_conf = 0.3

# ─── 비율 유지 리사이즈 + 패딩 함수 ───────────────────────────────────────
def resize_with_pad(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    img         : 원본 BGR 이미지 (h0, w0)
    new_shape   : 목표 출력 크기 (h, w)
    return      : (img_padded, ratio, (dw, dh))
    ratio       : resize 비율
    dw, dh      : 전체 패딩 너비, 높이 (양쪽 합)
    """
    h0, w0 = img.shape[:2]
    nh, nw = new_shape

    # 1) 비율 계산 (scale down/up 모두 허용)
    r = min(nh / h0, nw / w0)

    # 2) 리사이즈
    unpad_w, unpad_h = int(w0 * r), int(h0 * r)
    img_resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    # 3) 패딩 계산
    dw, dh = nw - unpad_w, nh - unpad_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    # 4) 패딩 추가
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return img_padded, r, (dw, dh)

# 교차(box) 겹침 판단
def box_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

# ─── 신뢰도 매핑 함수 ─────────────────────────────────────────
def map_confidence(conf, conf_thres=usr_conf):
    if conf < conf_thres:
        return conf
    if conf <= 0.6:
        return 0.90 + (conf - conf_thres) / (0.6 - conf_thres) * (0.92 - 0.90)
    elif conf <= 0.8:
        return 0.92 + (conf - 0.6) / 0.2 * (0.95 - 0.92)
    else:  
        return 0.95 + (min(conf, 1.0) - 0.8) / 0.2 * (0.97 - 0.95)

# ─── 웹캠 열기 ────────────────────────────────────────────────────────────
# cap = cv2.VideoCapture(0)
# #cap = cv2.VideoCapture(1)  # 다이소 웹캠
# if not cap.isOpened():
#     raise RuntimeError("카메라를 열 수 없습니다.")

print("---- 실시간 객체 인식 시작 (종료: 'q' 키) ----")

# ZED 카메라 시작
import uuid

def check_device(model_name):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    sub_model_paths = {
        'BK20S-T2':     'runs/small/BK20_v4.pt',
        'BS32c-6A':     'runs/small/BS32_v4.pt',
        'WAGO750-1505': 'runs/small/wago_v4.pt',
    }
    if model_name not in sub_model_paths:
        raise ValueError(f"모델 '{model_name}' 은(는) 존재하지 않습니다.")

    model_path = sub_model_paths[model_name]
    model = DetectMultiBackend(model_path, device=device)
    class_names = model.names

    # Stop threshold depends on model
    if model_name == 'WAGO750-1505':
        stop_threshold = 32
    else:
        stop_threshold = 2

    with ZedCamera() as zed:
        print("ZED 카메라가 시작되었습니다.")

        for i in range(75):
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            img0 = frame.copy()
            h0, w0 = img0.shape[:2]

            # 밝은 부분만 블러링
            gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            bright_mask_3ch = cv2.merge([bright_mask] * 3)
            blurred = cv2.GaussianBlur(img0, (5, 5), 0)
            img0 = np.where(bright_mask_3ch == 255, blurred, img0)

            # 전처리
            img1, ratio, (dw, dh) = resize_with_pad(img0, new_shape=(640, 640))
            img = img1[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0

            # 추론
            with torch.no_grad():
                pred = model(img_tensor, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.45)[0]

            display_img = img0.copy()
            connect_count = 0
            disconnect_count = 0
            confidence_dict = {
                "connect": [],
                "disconnect": []
            }

            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes((640, 640), pred[:, :4], (h0, w0)).round()
                annotator = Annotator(display_img, line_width=2, example=str(class_names))

                for *xyxy, conf, cls in pred:
                    label = class_names[int(cls)]
                    mapped_conf = map_confidence(float(conf), conf_thres=usr_conf)
                    label_text = f"{mapped_conf:.2f}"
                    color = (0, 255, 0) if label == 'connect' else (0, 0, 255)
                    annotator.box_label(xyxy, label_text, color=color)

                    if label == 'connect':
                        connect_count += 1
                        confidence_dict["connect"].append(round(mapped_conf, 2))
                    elif label == 'disconnect':
                        disconnect_count += 1
                        confidence_dict["disconnect"].append(round(mapped_conf, 2))

                display_img = annotator.result()

            # Check stop condition (do not accumulate counts across frames)
            if (connect_count + disconnect_count) >= stop_threshold:
                image_name = f"{model_name.lower()}_{uuid.uuid4()}.jpg"
                return {
                "status": {
                    "connect": connect_count,
                    "disconnect": disconnect_count
                },
                "confidence": confidence_dict,
                "image_name": image_name,
                "image": display_img
                }
            
            #print(connect_count + disconnect_count)
            print(i)

    return {"status": "failed"}
