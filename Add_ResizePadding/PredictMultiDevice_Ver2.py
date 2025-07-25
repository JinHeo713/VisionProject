import torch
import cv2
import numpy as np
from models.common     import DetectMultiBackend
from utils.general     import non_max_suppression, scale_boxes
from utils.plots       import Annotator
from utils.torch_utils import select_device

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
def map_confidence_wago(orig_pct):
    if conf <= 0.6 or conf > 0.9:
        return conf
    elif conf <= 0.7:
        return 0.82 + (conf - 0.6) / 0.1 * (0.90 - 0.82)
    elif conf <= 0.8:
        return 0.90 + (conf - 0.7) / 0.1 * (0.92 - 0.90)
    else:  # 0.8–0.9
        return 0.92 + (conf - 0.8) / 0.1 * (0.96 - 0.92)
    
def map_confidence(orig_pct):
    if conf <= 0.7 or conf > 0.9:
        return conf
    elif conf <= 0.75:
        return 0.80 + (conf - 0.7) / 0.05 * (0.85 - 0.80)
    elif conf <= 0.8:
        return 0.85 + (conf - 0.75) / 0.05 * (0.92 - 0.85)
    else:  
        return 0.92 + (conf - 0.8) / 0.10 * (0.96 - 0.92)

# ─── 모델 로드 ────────────────────────────────────────────────────────────
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model_device = DetectMultiBackend(
    "runs/train/connect_yolov5s_results/weights/Device_windows.pt",
    device=device
)
device_names = model_device.names

sub_model_paths = {
    'BK20S-T2':     'runs/train/connect_yolov5s_results/weights/BK20S_v3_windows.pt',
    'BS32c-6A':     'runs/train/connect_yolov5s_results/weights/BS32_v3_windows.pt',
    'WAGO750-1505': 'runs/train/connect_yolov5s_results/weights/wago_v3_windows.pt',
}
sub_models_loaded = {}

# ─── 웹캠 열기 ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1)  # 다이소 웹캠
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("---- 실시간 객체 인식 시작 (종료: 'q' 키) ----")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) 원본 복사
    img0 = frame.copy()
    h0, w0 = img0.shape[:2]

    # 2) 리사이즈 + 패딩 (비율 유지)
    img1, ratio, (dw, dh) = resize_with_pad(img0, new_shape=(640, 640))

    # 3) RGB→CHW, Tensor
    img = img1[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0

    # 4) 1차 추론
    with torch.no_grad():
        pred1 = model_device(img_tensor, augment=False, visualize=False)
        preds1 = non_max_suppression(pred1, conf_thres=0.6, iou_thres=0.45)
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
            #label    = f"{cls_name} {conf:.2f}"
            if cls_name == 'WAGO750-1505':                                
                mapped = map_confidence_wago(float(conf))                     
                label = f"{cls_name} {mapped:.2f}"                       
            else:
               label = f"{cls_name} {conf:.2f}"
            
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
                    preds2 = non_max_suppression(pred2, conf_thres=0.6, iou_thres=0.45)
                    pred2 = preds2[0] if len(preds2[0]) else None

                if pred2 is not None and len(pred2):
                    pred2[:, :4] = scale_boxes((640, 640), pred2[:, :4], (h0, w0)).round()
                    for *xyxy2, conf2, cls2 in pred2:
                        if not box_overlap(xyxy2, xyxy):
                            continue
                        cls2_name = sub_names[int(cls2)]

                        if cls_name == 'WAGO750-1505':                        
                            mapped_conf2 = map_confidence_wago(float(conf2))      
                        else:
                            mapped_conf2 = map_confidence(float(conf2))
                        label2 = f"{cls2_name} {mapped_conf2:.2f}"
                        color2 = (0,255,0) if cls2_name=='connect' else (0,0,255)
                        annotator.box_label(xyxy2, label2, color=color2)

        display_img = annotator.result()

    # 8) 화면 표시
    cv2.imshow("Real-time YOLO Detection", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
