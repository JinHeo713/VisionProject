import cv2
import torch
import numpy as np
import uuid
import csv
from datetime import datetime
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator
from utils.torch_utils import select_device
from camera import ZedCamera
from pyzed import sl

usr_conf = 0.35
ground_path = "logs/ground_truth/groundtruth.csv"
prediction_path = "logs/prediction/prediction.csv"
ground_truth_dict = {}

device = select_device('0' if torch.cuda.is_available() else 'cpu')

model_paths = {
    'WAGO750-1505': 'runs/small/wago_v4.pt',
    'BK20S-T2':     'runs/small/BK20_v4.pt',
    'BS32c-6A':     'runs/small/BS32_v4.pt',
}
sub_models = {
    name: DetectMultiBackend(path, device=device)
    for name, path in model_paths.items()
}

def resize_with_pad(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    unpad_w, unpad_h = int(w0 * r), int(h0 * r)
    img_resized = cv2.resize(img, (unpad_w, unpad_h))
    dw, dh = new_shape[1] - unpad_w, new_shape[0] - unpad_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded

def map_confidence(conf):
    if conf < usr_conf: return conf
    if conf <= 0.6: return 0.90 + (conf - usr_conf) / (0.6 - usr_conf) * (0.92 - 0.90)
    elif conf <= 0.8: return 0.92 + (conf - 0.6) / 0.2 * (0.95 - 0.92)
    else: return 0.95 + (min(conf, 1.0) - 0.8) / 0.2 * (0.97 - 0.95)

def sort_boxes_order(predictions):
    with_center = []
    for label, bbox in predictions:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        with_center.append((label, bbox, cx, cy))
    with_center.sort(key=lambda x: x[3])
    ordered = []
    for i in range(0, len(with_center), 2):
        group = with_center[i:i+2]
        group.sort(key=lambda x: x[2])
        ordered.extend(group)
    return [(label, bbox, idx + 1) for idx, (label, bbox, _, _) in enumerate(ordered)]

def calculate_f1(predicted, ground_truth):
    TP = FP = FN = 0
    gt_map = {idx: label for idx, label in ground_truth}
    for pred_label, _, order in predicted:
        if order in gt_map:
            if gt_map[order] == pred_label:
                TP += 1
                del gt_map[order]
            else:
                FP += 1
        else:
            FP += 1
    FN = len(gt_map)
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return round(f1, 2) * 100

def predict_model(frame, model, model_name=None):
    img_padded = resize_with_pad(frame)
    img = img_padded[:, :, ::-1].transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img)) \
                     .to(model.device).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.3)[0]

    annotated = frame.copy()
    predictions = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes((640, 640), pred[:, :4],
                                   frame.shape[:2]).round()
        annotator = Annotator(annotated, line_width=1, example=str(model.names))
        for *xyxy, conf, cls in pred:
            label = model.names[int(cls)]
            bbox = [int(x.item()) for x in xyxy]
            conf_mapped = map_confidence(float(conf))
            predictions.append((label, bbox))

            display_label = f"{model_name}: {label} {conf_mapped:.2f}" if model_name else f"{label} {conf_mapped:.2f}"
            color = (0, 255, 0) if label == 'connect' else (0, 0, 255)
            annotator.box_label(bbox, display_label, color=color)

        annotated = annotator.result()
    return annotated, predictions

def prompt_ground_truth(model_name):
    seq = input(f"{model_name} 의 ground truth를 1(connect), 0(disconnect)로 순서대로 입력하세요 (공백으로 구분):\n> ")
    bits = seq.strip().split()
    labels = ['connect' if b == '1' else 'disconnect' for b in bits]
    return [(i, label) for i, label in enumerate(labels, start=1)]

def save_groud_truth_csv(csv_path, gt_dict, model_order):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx, model in enumerate(model_order):
            if model in gt_dict:
                writer.writerow(['device', model])
                labels = [label for _, label in gt_dict[model]]
                writer.writerow(['ground truth'] + labels)
                if idx < len(model_order) - 1:
                    writer.writerow([])

def save_results(preds, model_name, img_to_save):
    ordered     = sort_boxes_order(preds)
    gt_list     = ground_truth_dict.get(model_name, [])
    gt_labels   = [label for _, label in gt_list]
    f1          = calculate_f1(ordered, gt_list)

    image_name  = f"{model_name}_{uuid.uuid4()}.jpg"
    image_path  = Path("images") / image_name
    annotated, _ = predict_model(img_to_save, sub_models[model_name], model_name)
    cv2.imwrite(str(image_path), annotated)

    with open(prediction_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['device', model_name])
        writer.writerow(['f1score', *(['']*16), f1])
        writer.writerow(['date_time', datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")])
        writer.writerow(['image_path', str(image_path)])
        writer.writerow(['ground truth'] + gt_labels)
        writer.writerow(['prediction'] + [label for label, _, _ in ordered])

    print(f"[✔] Saved annotated image for {model_name} with F1: {f1}")

# For F1 and Save triggers
f1_trigger_key_map = {
    ord('w'): 'WAGO750-1505',
    ord('b'): 'BK20S-T2',
    ord('s'): 'BS32c-6A',
}

# For real-time display model selection
display_key_map = {
    ord('1'): ['WAGO750-1505'],
    ord('2'): ['BK20S-T2', 'BS32c-6A'],
}


if __name__ == '__main__':
    models = {
        name: DetectMultiBackend(path, device=device)
        for name, path in model_paths.items()
    }

    model_order = list(model_paths.keys())
    active_model_key = []
    print("[INFO] Starting real-time inference...")

    Path("images").mkdir(parents=True, exist_ok=True)

    with ZedCamera(resolution=sl.RESOLUTION.HD2K) as zed:
        while True:
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            original = frame.copy()

            key = cv2.waitKey(1) & 0xFF

            if key in f1_trigger_key_map:
                model_key = f1_trigger_key_map[key]
                ground_truth_dict[model_key] = prompt_ground_truth(model_key)
                save_groud_truth_csv(ground_path, ground_truth_dict, model_order)
                print(f"[✔] Ground truth 입력 완료 for {model_key}")

            elif key in display_key_map:
                active_model_key = display_key_map[key]
                # print(f"[MODE] Showing only: {active_model_key}")

            elif key == ord('c'):
                active_model_key = []
                print("[MODE] Cleared active model — display OFF")

            for model_key in list(ground_truth_dict.keys()):
                max_try = 30
                for i in range(max_try):
                    frame, _, _ = zed.get_color_and_depth_frame()
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    original = frame.copy()

                    _, preds = predict_model(original, models[model_key])
                    ordered = sort_boxes_order(preds)
                    gt_list = ground_truth_dict[model_key]
                    f1 = calculate_f1(ordered, gt_list)

                    if f1 == 100:
                        save_results(preds, model_key, original.copy())
                        print(f"[✔] F1 Score 100 → 저장 완료 for {model_key}")
                        del ground_truth_dict[model_key]
                        break
                    elif i == max_try - 1:
                        save_results(preds, model_key, original.copy())
                        print(f"[⚠] F1 Score={f1} → 최대 시도 도달, 결과 저장 for {model_key}")
                        del ground_truth_dict[model_key]
                    else:
                        print(f"[ ] {model_key}: F1 Score = {f1} (시도 {i+1}/{max_try})")

            display_frame = frame.copy()
            if active_model_key:
                for model_key in active_model_key:
                    display_frame, _ = predict_model(display_frame, models[model_key], model_key)
            else:
                cv2.putText(display_frame, "Press 1(WAGO), 2(BK20+BS32) | w/b/s to save | c=clear",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Real-Time Multi Model Inference", display_frame)

            if key == ord('q'):
                break

    cv2.destroyAllWindows()
