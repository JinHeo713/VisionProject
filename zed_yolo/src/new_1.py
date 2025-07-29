# This is a consolidated and clean version of your code.
# Make sure required files and folders exist before running.

import torch
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator
from utils.torch_utils import select_device
from camera import ZedCamera
from pathlib import Path
from datetime import datetime
import csv
import uuid
import os
import threading


usr_conf = 0.3
ground_path = "logs/ground_truth/groundtruth.csv"
prediction_path = "logs/prediction/prediction.csv"

model_paths = {
    "1": ("WAGO750-1505", 'runs/small/wago_v4.pt'),
    "2": ("BK20S-T2", 'runs/small/BK20_v4.pt'),
    "3": ("BS32c-6A", 'runs/small/BS32_v4.pt')
}

def load_ground_truth(csv_path):
    ground_truth = {}
    current_device = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not any(row):
                continue
            if row[0].strip().lower() == 'device':
                current_device = row[1].strip()
                ground_truth[current_device] = []
            elif row[0].strip().lower() == 'ground truth' and current_device:
                for i, label in enumerate(row[1:], start=1):
                    label = label.strip().lower()
                    if label in ['connect', 'disconnect']:
                        ground_truth[current_device].append((i, label))
    return ground_truth

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
    return [(label, bbox, idx + 1) for idx, (label, bbox, cx, cy) in enumerate(ordered)]

def map_confidence(conf, conf_thres=usr_conf):
    if conf < conf_thres:
        return conf
    if conf <= 0.6:
        return 0.90 + (conf - conf_thres) / (0.6 - conf_thres) * (0.92 - 0.90)
    elif conf <= 0.8:
        return 0.92 + (conf - 0.6) / 0.2 * (0.95 - 0.92)
    else:
        return 0.95 + (min(conf, 1.0) - 0.8) / 0.2 * (0.97 - 0.95)

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
        'Precision': precision, 'Recall': recall, 'F1_Score': round(f1_score, 2)
    }

def run_detection(model_name, model_path, model, device, class_names):
    with ZedCamera() as zed:
        while True:
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            img0 = frame.copy()
            h0, w0 = img0.shape[:2]
            img = cv2.resize(img0, (640, 640))
            img_tensor = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1)).float().to(device) / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():
                pred = model(img_tensor, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.45)[0]

            predictions = []
            if pred is not None and len(pred):
                pred[:, :4] = scale_boxes((640, 640), pred[:, :4], (h0, w0)).round()
                annotator = Annotator(img0, line_width=2, example=str(class_names))
                for *xyxy, conf, cls in pred:
                    label = class_names[int(cls)]
                    bbox = [int(x) for x in xyxy]
                    predictions.append((label, bbox))
                ordered = sort_boxes_order(predictions)

                for label, bbox, order_idx in ordered:
                    text = f"{label} ({order_idx})"
                    color = (0, 255, 0) if label == 'connect' else (0, 0, 255)
                    annotator.box_label(bbox, text, color=color)

                GROUND_TRUTH = load_ground_truth(ground_path)
                if model_name in GROUND_TRUTH:
                    f1 = calculate_order_f1_metrics(ordered, GROUND_TRUTH[model_name])
                else:
                    f1 = {}

                cv2.imshow(f"Detection - {model_name}", annotator.result())

                # Save logs
                os.makedirs("images", exist_ok=True)
                os.makedirs(Path(prediction_path).parent, exist_ok=True)
                filename = f"{model_name.lower()}_{uuid.uuid4().hex[:8]}.jpg"
                img_path = f"images/{filename}"
                cv2.imwrite(img_path, img0)

                with open(prediction_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['device', model_name])
                    writer.writerow(['f1score', '', '', '', f1.get('F1_Score', '')])
                    writer.writerow(['datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    writer.writerow(['image', img_path])
                    gt_labels = [label for _, label in GROUND_TRUTH.get(model_name, [])]
                    pred_dict = {idx: label for label, _, idx in ordered}
                    pred_labels = [pred_dict.get(i, "") for i in range(1, len(gt_labels)+1)]
                    writer.writerow(['ground truth'] + gt_labels)
                    writer.writerow(['prediction'] + pred_labels)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        cv2.destroyAllWindows()

# def main():
#     device = select_device('0' if torch.cuda.is_available() else 'cpu')
#     print("Press 1 (WAGO), 2 (BK20), 3 (BS32), ESC to exit")
#     while True:
#         key = cv2.waitKey(0)
#         if key == 27:  # ESC
#             break

#         if key in [ord('1'), ord('2'), ord('3')]:
#             key_char = chr(key)
#             model_name, model_path = model_paths[key_char]
#             print(f"Loading model: {model_name}")
#             model = DetectMultiBackend(model_path, device=device)
#             run_detection(model_name, model_path, model, device, model.names)
#         else:
#             print("Invalid key. Press 1 (WAGO), 2 (BK20), 3 (BS32), ESC to exit.")

# 실시간 피드 제어 플래그
stop_flag = False

def live_display():
    """ZedCamera로부터 프레임을 받아 실시간으로 띄워줍니다."""
    with ZedCamera() as zed:
        while not stop_flag:
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow("Live Feed", frame)
            # 'q' 누르면 실시간 창만 닫고 루프 빠져나갑니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyWindow("Live Feed")


if __name__ == "__main__":
    # 1) 실시간 디스플레이 스레드 시작
    t = threading.Thread(target=live_display, daemon=True)
    t.start()

    # 2) 입력 ↔ 모델명 매핑
    shortcut_map = {
        'wago':  'WAGO750-1505',
        'bk20s': 'BK20S-T2',
        'bs32c': 'BS32c-6A',
    }

    print("실시간 피드가 켜졌습니다. 모델 키워드를 입력하세요.")
    print("종료하려면 'exit' 입력 또는 실시간 창에서 'q' 누르기")

    while True:
        choice = input("모델 (wago / bk20s / bs32c / exit): ").strip().lower()

        if choice == 'exit':
            stop_flag = True
            break

        if choice not in shortcut_map:
            print(f"잘못된 입력: '{choice}'. 다시 입력하세요.")
            continue

        model_name = shortcut_map[choice]
        print(f">>> {model_name} 모델 실행 중...")

        # 한 번 실행하고 CSV 기록
        result = check_device(model_name)

        # 실행 결과 요약
        print(f"[{model_name}] connect: {result['status']['connect']}  disconnect: {result['status']['disconnect']}  F1: {result['f1_score']}")
        print(f"CSV: {result['errata_csv']}  이미지: images/{result['image_name']}")

        # 스냅샷 보기
        cv2.imshow(f"{model_name} Snapshot", result['image'])
        cv2.waitKey(1)  # 창만 띄우고 바로 리턴
        cv2.destroyWindow(f"{model_name} Snapshot")

    # 3) 깔끔히 종료
    t.join()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")
    # main()