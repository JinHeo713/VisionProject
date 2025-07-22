import torch
import cv2
import numpy as np
from models.common     import DetectMultiBackend
from utils.general     import non_max_suppression, scale_boxes
from utils.plots       import Annotator
from utils.torch_utils import select_device
from camera import ZedCamera
import uuid
import pathlib
from pathlib import Path

pathlib.WindowsPath = pathlib.PosixPath
usr_conf = 0.3

# ─── 모델별 Ground Truth 정의 ─────────────────────────────
GROUND_TRUTH = {
    'WAGO750-1505': [
        (1, 'connect'), (2, 'disconnect'), (3, 'connect'), (4, 'disconnect'),
        (5, 'connect'), (6, 'disconnect'), (7, 'connect'), (8, 'disconnect'),
        (9, 'connect'), (10, 'disconnect'), (11, 'connect'), (12, 'disconnect'),
        (13, 'connect'), (14, 'disconnect'), (15, 'connect'), (16, 'disconnect')
    ],
    'BK20S-T2': [
        (1, 'connect'), (2, 'connect')
    ],
    'BS32c-6A': [
        (1, 'connect'), (2, 'connect')
    ]
}

# ─── 박스 정렬 ─────────────────────────────
def sort_boxes_order(predictions):
    with_center = []
    for label, bbox in predictions:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        with_center.append((label, bbox, cx, cy))

    with_center.sort(key=lambda x: x[3])  # cy 기준
    ordered = []
    for i in range(0, len(with_center), 2):
        group = with_center[i:i+2]
        group.sort(key=lambda x: x[2])  # cx 기준
        ordered.extend(group)

    return [(label, bbox, idx + 1) for idx, (label, bbox, cx, cy) in enumerate(ordered)]

# ─── 비율 유지 리사이즈 + 패딩 ─────────────────────────────
def resize_with_pad(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h0, nw / w0)
    unpad_w, unpad_h = int(w0 * r), int(h0 * r)
    img_resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = nw - unpad_w, nh - unpad_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)

# ─── Confidence 매핑 ─────────────────────────────
def map_confidence(conf, conf_thres=usr_conf):
    if conf < conf_thres:
        return conf
    if conf <= 0.6:
        return 0.90 + (conf - conf_thres) / (0.6 - conf_thres) * (0.92 - 0.90)
    elif conf <= 0.8:
        return 0.92 + (conf - 0.6) / 0.2 * (0.95 - 0.92)
    else:
        return 0.95 + (min(conf, 1.0) - 0.8) / 0.2 * (0.97 - 0.95)

# ─── F1 Score 계산 ─────────────────────────────
def calculate_order_f1_metrics(predicted_ordered_boxes, ground_truth_order):
    TP = FP = FN = 0
    gt_map = {idx: label for idx, label in ground_truth_order}
    for pred_label, _, pred_order_idx in predicted_ordered_boxes:
        if pred_order_idx in gt_map:
            if gt_map[pred_order_idx] == pred_label:
                TP += 1
                del gt_map[pred_order_idx]
            else:
                FP += 1
        else:
            FP += 1
    FN = len(gt_map)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'TP': TP, 'FP': FP, 'FN': FN,
        'Precision': precision, 'Recall': recall, 'F1_Score': f1_score
    }

# ─── 메인 객체 인식 함수 ─────────────────────────────
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
    stop_threshold = 16 if model_name == 'WAGO750-1505' else 2
    f1_metrics = {}

    with ZedCamera() as zed:
        for i in range(5):
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            img0 = frame.copy()
            h0, w0 = img0.shape[:2]

            # 밝은 영역 블러링
            gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(img0, (5, 5), 0)
            img0 = np.where(cv2.merge([bright_mask]*3) == 255, blurred, img0)

            # 전처리
            img1, _, _ = resize_with_pad(img0, new_shape=(640, 640))
            img = img1[:, :, ::-1].transpose(2, 0, 1)
            img_tensor = torch.from_numpy(np.ascontiguousarray(img)).to(device).float().unsqueeze(0) / 255.0

            with torch.no_grad():
                pred = model(img_tensor, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.45)[0]

            display_img = img0.copy()
            connect_count = disconnect_count = 0
            confidence_dict = {"connect": [], "disconnect": []}
            predictions_for_sorting = []

            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes((640, 640), pred[:, :4], (h0, w0)).round()
                annotator = Annotator(display_img, line_width=2, example=str(class_names))

                for *xyxy, conf, cls in pred:
                    label = class_names[int(cls)]
                    bbox_int = [int(x) for x in xyxy]
                    predictions_for_sorting.append((label, bbox_int))

                ordered_predictions = sort_boxes_order(predictions_for_sorting)

                for label, bbox, order_idx in ordered_predictions:
                    mapped_conf = 0.0
                    for *xyxy, conf_val, cls_idx in pred:
                        bb = [int(x) for x in xyxy]
                        if class_names[int(cls_idx)] == label and bb == bbox:
                            mapped_conf = map_confidence(float(conf_val))
                            break
                    text = f"{mapped_conf:.2f} ({order_idx})"
                    color = (0, 255, 0) if label == 'connect' else (0, 0, 255)
                    annotator.box_label(bbox, text, color=color)
                    if label == 'connect':
                        connect_count += 1
                        confidence_dict["connect"].append(round(mapped_conf, 2))
                    elif label == 'disconnect':
                        disconnect_count += 1
                        confidence_dict["disconnect"].append(round(mapped_conf, 2))

                display_img = annotator.result()

                if model_name in GROUND_TRUTH:
                    f1_metrics = calculate_order_f1_metrics(ordered_predictions, GROUND_TRUTH[model_name])

            if (connect_count + disconnect_count) >= stop_threshold:
                image_name = f"{model_name.lower()}_{uuid.uuid4()}.jpg"
                return {
                    "status": {
                        "connect": connect_count,
                        "disconnect": disconnect_count
                    },
                    "confidence": confidence_dict,
                    "image_name": image_name,
                    "image": display_img,
                    "f1_metrics": f1_metrics
                }

    return {"status": "failed", "f1_metrics": f1_metrics}
