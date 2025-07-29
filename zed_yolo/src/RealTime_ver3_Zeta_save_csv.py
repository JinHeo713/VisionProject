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

usr_conf = 0.3
ground_path = "logs/ground_truth/groundtruth.csv"
prediction_path = "logs/prediction/prediction.csv"

model_paths = {
    'WAGO750-1505': 'runs/small/wago_v4.pt',
    'BK20S-T2':     'runs/small/BK20_v4.pt',
    'BS32c-6A':     'runs/small/BS32_v4.pt',
}

# ---------- Util functions ----------
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

def load_ground_truth(csv_path):
    ground_truth = {}
    current_device = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not any(row): continue
            if row[0].strip().lower() == 'device':
                current_device = row[1].strip()
                ground_truth[current_device] = []
            elif row[0].strip().lower() == 'ground truth' and current_device:
                for i, label in enumerate(row[1:], start=1):
                    if label.strip().lower() in ['connect', 'disconnect']:
                        ground_truth[current_device].append((i, label.strip().lower()))
    return ground_truth

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
    return round(f1, 2)

# ---------- Model prediction ----------
def predict_model(frame, model, name):
    img_padded = resize_with_pad(frame)
    img = img_padded[:, :, ::-1].transpose(2, 0, 1)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img)).to(model.device).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.45)[0]

    annotated = frame.copy()
    predictions = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_boxes((640, 640), pred[:, :4], frame.shape[:2]).round()
        annotator = Annotator(annotated, line_width=2, example=str(model.names))
        for *xyxy, conf, cls in pred:
            label = model.names[int(cls)]
            bbox = [int(x.item()) for x in xyxy]
            conf_mapped = map_confidence(float(conf))
            predictions.append((label, bbox))
            color = (0, 255, 0) if label == 'connect' else (0, 0, 255)
            annotator.box_label(bbox, f"{label} {conf_mapped:.2f}", color=color)
        annotated = annotator.result()
    return annotated, predictions

def save_results(preds, model_name, img_to_save):
    ordered    = sort_boxes_order(preds)
    GROUND_TRUTH = load_ground_truth(ground_path)
    gt_labels  = [label for _, label in GROUND_TRUTH.get(model_name, [])]

    max_idx    = max(max((o for _, _, o in ordered), default=0), len(gt_labels))
    pred_dict  = {order: label for label, _, order in ordered}
    pred_labels = [pred_dict.get(i, '') for i in range(1, max_idx + 1)]

    f1         = calculate_f1(ordered, GROUND_TRUTH.get(model_name, []))
    image_name = f"{model_name}_{uuid.uuid4()}.jpg"
    image_path = Path("images") / image_name

    # 여기서 img_to_save(바운딩 박스 그려진 이미지)를 저장
    cv2.imwrite(str(image_path), img_to_save)

    with open(prediction_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['device', model_name])
        writer.writerow(['f1score', *(['']*16), f1])
        writer.writerow(['date_time', datetime.now().strftime("%Y_%m_%d_%H_%M_%S")])
        writer.writerow(['image_path', str(image_path)])
        writer.writerow(['ground truth'] + gt_labels)
        writer.writerow(['prediction'] + pred_labels)
    print(f"[✔] Saved annotated image for {model_name} with F1: {f1}")    

# ---------- Main ----------
if __name__ == '__main__':
    print("=== Writing ground truth ===")
    # ground_path 변수는 이미 선언되어 있다고 가정
    with open(ground_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 모델 순서대로 입력
        for model_name in ['WAGO750-1505', 'BK20S-T2', 'BS32c-6A']:
            # 1이면 connect, 0이면 disconnect
            seq = input(f"{model_name} 의 ground truth를 1(connect), 0(disconnect)로 순서대로 입력하세요 (공백으로 구분):\n> ")
            bits = seq.strip().split()
            labels = ['connect' if b=='1' else 'disconnect' for b in bits]
            # CSV에 쓰기
            writer.writerow(['device', model_name])
            writer.writerow(['ground truth'] + labels)

    print(f"Ground truth saved to {ground_path}\n")

    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    models = {name: DetectMultiBackend(path, device=device) for name, path in model_paths.items()}
    # cap = cv2.VideoCapture(1)

    # while True:
    #     # 1) ZED에서 프레임 읽기
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Fail to read Camera")
    #         break
    #     # 2) 왼쪽 렌즈(이미지)의 절반만 잘라내기
    #     h0, w0 = frame.shape[:2]
    #     left = frame[:, :w0//2]

    #     # 3) 밝은 영역 블러핑
    #     gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    #     _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    #     blurred = cv2.GaussianBlur(left, (5, 5), 0)
    #     original = np.where(cv2.merge([bright_mask]*3) == 255, blurred, left)

    #     # 4) 모델별 순차 예측 & 누적 어노테이션
    #     stacked_frame = original.copy()
    #     for name, model in models.items():
    #         stacked_frame, _ = predict_model(stacked_frame, model, name)

    #     cv2.imshow("🔍 Real-Time Multi Model Inference", stacked_frame)

    #     # 5) 키 입력 처리 (저장 기능)
    #     key = cv2.waitKey(1)
    #     if key == ord('w'):











    #         annotated, preds = predict_model(original, models['WAGO750-1505'], 'WAGO750-1505')
    #         save_results(preds, 'WAGO750-1505', annotated)  
    

    #     elif key == ord('b'):
    #         annotated, preds = predict_model(original, models['BK20S-T2'], 'BK20S-T2')
    #         save_results(preds, 'BK20S-T2', annotated)
    #     elif key == ord('s'):
    #         annotated, preds = predict_model(original, models['BS32c-6A'], 'BS32c-6A')
    #         save_results(preds, 'BS32c-6A', annotated)
    #     elif key == ord('q'):
    #         break
        
    # cap.release()
    # cv2.destroyAllWindows()

# ------------------제타 Zed 연결-------------------- #
    with ZedCamera() as zed:
        while True:
            # 1) ZED에서 프레임 읽기
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 2) 왼쪽 렌즈(이미지)의 절반만 잘라내기
            img0 = frame.copy()
            h0, w0 = img0.shape[:2]
            # left = frame[:, :w0//2]

            # 3) 밝은 영역 블러핑
            gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(img0, (5, 5), 0)
            original = np.where(cv2.merge([bright_mask]*3) == 255, blurred, img0)

            # 4) 모델별 순차 예측 & 누적 어노테이션
            stacked_frame = original.copy()
            for name, model in models.items():
                stacked_frame, _ = predict_model(stacked_frame, model, name)

            cv2.imshow("🔍 Real-Time Multi Model Inference", stacked_frame)

            # 5) 키 입력 처리 (저장 기능)
            key = cv2.waitKey(1)
            if key == ord('w'):
                annotated, preds = predict_model(original, models['WAGO750-1505'], 'WAGO750-1505')
                save_results(preds, 'WAGO750-1505', annotated)
            elif key == ord('b'):
                annotated, preds = predict_model(original, models['BK20S-T2'], 'BK20S-T2')
                save_results(preds, 'BK20S-T2', annotated)
            elif key == ord('s'):
                annotated, preds = predict_model(original, models['BS32c-6A'], 'BS32c-6A')
                save_results(preds, 'BS32c-6A', annotated)
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()

#-------------------------------------------------------------------------------#
#----------------------키 입력마다 groundtruth.csv파일 다시 쓰기-------------------#

# ─── Ground Truth 업데이트 헬퍼 ─────────────────────────────
# def save_ground_truth_entry(model_name):
#     # 1이면 connect, 0이면 disconnect 입력받기
#     seq = input(f"{model_name} 의 ground truth를 1(connect), 0(disconnect)로 순서대로 입력하세요 (공백 구분):\n> ")
#     bits = seq.strip().split()
#     labels = ['connect' if b=='1' else 'disconnect' for b in bits]

#     # 기존 ground truth 불러와서 해당 모델만 교체
#     gt = load_ground_truth(ground_path)
#     gt[model_name] = [(i+1, lbl) for i, lbl in enumerate(labels)]

#     # 전체 모델 순서대로 CSV 덮어쓰기
#     with open(ground_path, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         for m in ['WAGO750-1505', 'BK20S-T2', 'BS32c-6A']:
#             writer.writerow(['device', m])
#             # 아직 입력된 게 없으면 빈 리스트 처리
#             m_labels = [lbl for _, lbl in gt.get(m, [])]
#             writer.writerow(['ground truth'] + m_labels)

#     print(f"[✔] {model_name} ground truth updated: {labels}")

# # ---------- Save CSV + Image ----------
# def save_results(preds, model_name, frame):
#     ordered = sort_boxes_order(preds)
#     GROUND_TRUTH = load_ground_truth(ground_path)
#     gt_labels = [label for _, label in GROUND_TRUTH.get(model_name, [])]

#     max_idx = max(max((o for _, _, o in ordered), default=0), len(gt_labels))
#     pred_dict = {order: label for label, _, order in ordered}
#     pred_labels = [pred_dict.get(i, '') for i in range(1, max_idx + 1)]

#     f1 = calculate_f1(ordered, GROUND_TRUTH.get(model_name, []))
#     image_name = f"{model_name}_{uuid.uuid4()}.jpg"
#     image_path = f"images/{image_name}"
#     cv2.imwrite(image_path, frame)

#     with open(prediction_path, 'a', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['device', model_name])
#         writer.writerow(['f1score', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', f1])
#         writer.writerow(['date_time', datetime.now().strftime("%Y_%m_%d_%H_%M_%S")])
#         writer.writerow(['image_path', image_path])
#         writer.writerow(['ground truth'] + gt_labels)
#         writer.writerow(['prediction'] + pred_labels)
#     print(f"[✔] Saved result for {model_name} with F1: {f1}")

# if __name__ == '__main__':

#     device = select_device('0' if torch.cuda.is_available() else 'cpu')
#     models = {name: DetectMultiBackend(path, device=device) for name, path in model_paths.items()}

#     cap = cv2.VideoCapture(1)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Fail to read Camera")
#             break

#         # 모델별 예측 결과를 합성
#         stacked_frame = frame.copy()
#         for name, model in models.items():
#             pred_img, _ = predict_model(frame, model, name)
#             stacked_frame = cv2.addWeighted(stacked_frame, 0.5, pred_img, 0.5, 0)

#         cv2.imshow("🔍 Real-Time Multi Model Inference", stacked_frame)


#         key = cv2.waitKey(1)
#         if key == ord('w'):
#             save_ground_truth_entry('WAGO750-1505')
#             _, preds = predict_model(frame, models['WAGO750-1505'], 'WAGO750-1505')
#             save_results(preds, 'WAGO750-1505', frame)
#         elif key == ord('b'):
#             save_ground_truth_entry('BK20S-T2')
#             _, preds = predict_model(frame, models['BK20S-T2'], 'BK20S-T2')
#             save_results(preds, 'BK20S-T2', frame)
#         elif key == ord('s'):
#             save_ground_truth_entry('BS32c-6A')
#             _, preds = predict_model(frame, models['BS32c-6A'], 'BS32c-6A')
#             save_results(preds, 'BS32c-6A', frame)
#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()