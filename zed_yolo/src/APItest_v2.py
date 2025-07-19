import torch
import cv2
import numpy as np
import time
from collections import Counter
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from camera import ZedCamera


def resize_with_pad(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h0, nw / w0)
    unpad_w, unpad_h = int(w0 * r), int(h0 * r)
    img_resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = nw - unpad_w, nh - unpad_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)

def postprocess_result(label_list, model_name):
    counter = Counter(label_list)
    c = counter.get('connect', 0)
    d = counter.get('disconnect', 0)
    if model_name in ['BK20S-T2', 'BS32c-6A']:
        total = c + d
        if total == 0:
            return {'connect': 1, 'disconnect': 1}
        ratio_c = c / total
        c_fixed = round(ratio_c * 2)
        d_fixed = 2 - c_fixed
        return {'connect': c_fixed, 'disconnect': d_fixed}
    else:
        return {'connect': c, 'disconnect': d}


def check_device(input_device):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    model_paths = {
        'BK20S-T2':     'runs/small/BK20_v4.pt',
        'BS32c-6A':     'runs/small/BS32_v4.pt',
        'WAGO750-1505': 'runs/small/wago_v4.pt',
    }
    alias_map = {
        'wago': 'WAGO750-1505',
        'bk20': 'BK20S-T2',
        'bs32': 'BS32c-6A'
    }

    models = {}

    with ZedCamera() as zed:
        print("ZED 카메라가 시작되었습니다.")

        user_input = input_device
        if user_input == 'exit':
            return None, None
        if user_input not in alias_map:
            print("❌ 잘못된 모델명입니다.")
            return None, None

        model_name = alias_map[user_input]

        if model_name not in models:
            print(f"📦 모델 {model_name} 로딩 중...")
            model = DetectMultiBackend(model_paths[model_name], device=device)
            models[model_name] = (model, model.names)
        else:
            model, _ = models[model_name]

        print(f"🔍 {model_name} 모델로 예측 수행 중...")

        label_results = []
        boxes_xywh = []

        if not zed.is_frame_available():
            return None, None
        frame, _, _ = zed.get_color_and_depth_frame()

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        img0 = frame.copy()
        h0, w0 = img0.shape[:2]
        img1, ratio, (dw, dh) = resize_with_pad(img0, new_shape=(640, 640))
        img = img1[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = model(img_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes((640, 640), pred[:, :4], (h0, w0)).round()
            _, names = models[model_name]

            for *xyxy, conf, cls in pred:
                label = names[int(cls)]
                label_results.append(label)

                # Convert from xyxy to xywh
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                boxes_xywh.append({
                    'label': label,
                    'confidence': float(conf),
                    'xywh': [float(cx), float(cy), float(w), float(h)]
                })

        result = postprocess_result(label_results, model_name)
        print(f"✅ 최종 예측 결과: {result}\n")

        return result, boxes_xywh

        
def draw_boxes(img, boxes, class_names):
    for *xyxy, conf, cls in boxes:
        label = f"{class_names[int(cls)]} {conf:.2f}"
        xyxy = [int(x) for x in xyxy]
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return img
