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
usr_thr = 0.3

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
def map_confidence(conf, conf_thres=usr_thr):
    if conf < conf_thres:
        return conf
    if conf <= 0.6:
        return 0.90 + (conf - conf_thres) / (0.6 - conf_thres) * (0.92 - 0.90)
    elif conf <= 0.8:
        return 0.92 + (conf - 0.6) / 0.2 * (0.95 - 0.92)
    else:  
        return 0.95 + (min(conf, 1.0) - 0.8) / 0.2 * (0.97 - 0.95)

# ─── 모델 로드 ────────────────────────────────────────────────────────────
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model_device = DetectMultiBackend(
    "runs/Device.pt",
    device=device
)
device_names = model_device.names

sub_model_paths = {
    'BK20S-T2':     'runs/small/BK20_v4.pt',
    'BS32c-6A':     'runs/small/BS32_v4.pt',
    'WAGO750-1505': 'runs/small/wago_v4.pt',
}
sub_models_loaded = {}

# ─── 웹캠 열기 ────────────────────────────────────────────────────────────
# cap = cv2.VideoCapture(0)
# #cap = cv2.VideoCapture(1)  # 다이소 웹캠
# if not cap.isOpened():
#     raise RuntimeError("카메라를 열 수 없습니다.")

print("---- 실시간 객체 인식 시작 (종료: 'q' 키) ----")
# ZED 카메라 시작
with ZedCamera() as zed:
    print("ZED 카메라가 시작되었습니다.")

    while True:
        #color_frame, depth_frame, timestamp
        frame, dep_frame, timestamp = zed.get_color_and_depth_frame()
       
        # BGRA → BGR (3채널로 변환)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if not True: #ret
            break

        # 1) 원본 복사
        img0 = frame.copy()
        h0, w0 = img0.shape[:2]


        # 1-1) 이미지 → 그레이스케일 → 밝은 영역 마스크 생성
        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # 밝은 부분만 마스크 (값 > 200)

        # 1-2) 마스크를 확장해서 BGR 3채널과 매칭
        bright_mask_3ch = cv2.merge([bright_mask] * 3)

        # 1-3) 밝은 부분만 블러링
        blurred = cv2.GaussianBlur(img0, (5, 5), 0)

        # 1-4) 밝은 부분에만 블러 적용
        img0 = np.where(bright_mask_3ch == 255, blurred, img0)


        # 2) 리사이즈 + 패딩 (비율 유지)
        img1, ratio, (dw, dh) = resize_with_pad(img0, new_shape=(640, 640))

        # 3) RGB→CHW, Tensor
        img = img1[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0

        # 4) 1차 추론
        with torch.no_grad():
            pred1 = model_device(img_tensor, augment=False, visualize=False)
            preds1 = non_max_suppression(pred1, conf_thres=usr_thr, iou_thres=usr_thr)
            pred1 = preds1[0] if len(preds1[0]) else None

        # 5) 결과를 그릴 베이스 이미지
        display_img = img0.copy()

        if pred1 is not None and len(pred1):
            # 6) 좌표 역변환: (640,640)→(h0,w0)
            pred1[:, :4] = scale_boxes((640, 640), pred1[:, :4], (h0, w0)).round()

            # 7) Annotator로 fresh canvas 에만 그리기
            annotator = Annotator(display_img, line_width=2, example=str(device_names))

            # 1차 박스
            for *xyxy, conf, cls in pred1:
                cls_name = device_names[int(cls)]
                mapped_conf1 = map_confidence(conf, conf_thres=usr_thr)
                label    = f"{cls_name} {mapped_conf1:.2f}"
                
                color = (255,255,255)
                if cls_name == 'WAGO750-1505': color = (255, 0, 0)
                elif cls_name == 'BK20S-T2':   color = (0, 255, 255)
                elif cls_name == 'BS32c-6A':   color = (0, 165, 255)
                annotator.box_label(xyxy, label, color=color)

                # 2차 추론 (겹치는 경우만)
                if cls_name in sub_model_paths:
                    if cls_name not in sub_models_loaded:
                        m = DetectMultiBackend(sub_model_paths[cls_name], device=device)
                        sub_models_loaded[cls_name] = (m, m.names)
                    model_sub, sub_names = sub_models_loaded[cls_name]
                    with torch.no_grad():
                        pred2 = model_sub(img_tensor, augment=False, visualize=False)
                        preds2 = non_max_suppression(pred2, conf_thres=usr_thr, iou_thres=usr_thr)
                        pred2 = preds2[0] if len(preds2[0]) else None

                    if pred2 is not None and len(pred2):
                        pred2[:, :4] = scale_boxes((640, 640), pred2[:, :4], (h0, w0)).round()
                        for *xyxy2, conf2, cls2 in pred2:
                            if not box_overlap(xyxy2, xyxy):
                                continue
                            cls2_name = sub_names[int(cls2)]
                            mapped_conf2 = map_confidence(float(conf2), conf_thres=usr_thr)
                            label2 = f"{mapped_conf2:.2f}"
                            color2 = (0,255,0) if cls2_name=='connect' else (0,0,255)
                            annotator.box_label(xyxy2, label2, color=color2)

            display_img = annotator.result()

        # 8) 화면 표시
        cv2.imshow("Real-time YOLO Detection", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    # cap.release()
    cv2.destroyAllWindows()
