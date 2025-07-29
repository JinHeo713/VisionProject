import torch
import cv2
import numpy as np
from models.common     import DetectMultiBackend
from utils.general     import non_max_suppression, scale_boxes
from utils.plots       import Annotator
from utils.torch_utils import select_device
from camera import ZedCamera
import uuid
from pathlib import Path
import csv
from datetime import datetime

usr_conf = 0.3

# ─── 모델별 Ground Truth 정의 ─────────────────────────────'
prediction_path = "logs/prediction/prediction.csv"
ground_path = "logs/ground_truth/groundtruth.csv"

sub_model_paths = {
    'BK20S-T2': 'runs/small/BK20_v4.pt',
    'BS32c-6A': 'runs/small/BS32_v4.pt',
    'WAGO750-1505': 'runs/small/wago_v4.pt',
}

# ─── csv to dict─────────────────────────────'
def load_ground_truth(csv_path):
    ground_truth = {}
    current_device = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not any(row):
                continue  # skip empty rows

            if row[0].strip().lower() == 'device':
                current_device = row[1].strip()
                ground_truth[current_device] = []

            elif row[0].strip().lower() == 'ground truth' and current_device:
                for i, label in enumerate(row[1:], start=1):
                    label = label.strip().lower()
                    if label in ['connect', 'disconnect']:
                        ground_truth[current_device].append((i, label))

    return ground_truth

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
        'Precision': precision, 'Recall': recall, 'F1_Score': round(f1_score, 2)
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
    errata_log = []
    connect_count = disconnect_count = 0
    confidence_dict = {"connect": [], "disconnect": []}
    display_img = None  # 초기화
    ordered_predictions = []

    with ZedCamera() as zed:
        for _ in range(50):
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
                    Annotator(display_img).box_label(bbox, text, color=color)
                    if label == 'connect':
                        connect_count += 1
                        confidence_dict["connect"].append(round(mapped_conf, 2))
                    elif label == 'disconnect':
                        disconnect_count += 1
                        confidence_dict["disconnect"].append(round(mapped_conf, 2))

                display_img = annotator.result()
                GROUND_TRUTH = load_ground_truth(ground_path)

                # F1 계산 및 오류 로그 작성
                if model_name in GROUND_TRUTH:
                    ground_truth_order = GROUND_TRUTH[model_name]
                    f1_metrics = calculate_order_f1_metrics(ordered_predictions, ground_truth_order)

                    gt_map = {idx: label for idx, label in ground_truth_order}
                    pred_map = {idx: label for label, _, idx in ordered_predictions}
                    all_indices = set(gt_map.keys()).union(pred_map.keys())
                    for idx in sorted(all_indices):
                        gt_label = gt_map.get(idx)
                        pred_label = pred_map.get(idx)
                        if gt_label == pred_label:
                            match_status = "TP"
                        elif pred_label and gt_label:
                            match_status = "FP"
                        elif pred_label and not gt_label:
                            match_status = "FP"
                        elif not pred_label and gt_label:
                            match_status = "FN"
                        else:
                            continue
                        errata_log.append({
                            "order_idx": idx,
                            "predicted_label": pred_label or "",
                            "ground_truth_label": gt_label or "",
                            "match_status": match_status
                        })

            # Save results to ONE existing CSV file by appending rows in new format:
            csv_path = Path(prediction_path)
            csv_path.parent.mkdir(exist_ok=True, parents=True)

            date_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            image_name = f"{model_name.lower()}_{uuid.uuid4()}.jpg"
            image_path_str = f"images/{image_name}"

            # Prepare ground truth and prediction labels for CSV row
            gt_labels = [label for _, label in GROUND_TRUTH.get(model_name, [])]

            max_idx = max(max((order_idx for _, _, order_idx in ordered_predictions), default=0), len(gt_labels))
            pred_dict = {order_idx: label for label, _, order_idx in ordered_predictions}
            pred_labels = [pred_dict.get(i, "") for i in range(1, max_idx + 1)]


            # Append to existing CSV file
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['device', model_name.lower()])
                writer.writerow(['f1score', '','','','','','','','','','','','','','','','',f1_metrics.get('F1_Score', '')])
                writer.writerow(['date_time', date_time_str])
                writer.writerow(['image_path', image_path_str])
                writer.writerow(['ground truth'] + gt_labels)
                writer.writerow(['prediction'] + pred_labels)

            result = {
                "status": {
                    "connect": connect_count,
                    "disconnect": disconnect_count
                },
                "confidence": confidence_dict,
                "image_name": image_name,
                "image": display_img,
                "f1_score": f1_metrics.get("F1_Score", 0),
                "errata_csv": str(csv_path)
            }

            if (connect_count + disconnect_count) == stop_threshold:
                return result

            return result
    
def realtime_view_with_csv_logging():
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    ground_truth_data = load_ground_truth(ground_path)

    # 모델 로딩
    models = {}
    colors = {
        'BK20S-T2': (0, 255, 0),
        'BS32c-6A': (255, 0, 0),
        'WAGO750-1505': (0, 0, 255),
    }

    for model_name, path in sub_model_paths.items():
        model = DetectMultiBackend(path, device=device)
        model.warmup(imgsz=(1, 3, 640, 640))
        models[model_name] = {
            'model': model,
            'class_names': model.names,
            'color': colors[model_name]
        }

    with ZedCamera() as zed:
        while True:
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            original = frame.copy()
            h0, w0 = original.shape[:2]

            # 밝은 영역 블러링
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            blurred = cv2.GaussianBlur(original, (5, 5), 0)
            original = np.where(cv2.merge([bright_mask] * 3) == 255, blurred, original)

            annotated = original.copy()

            for model_name, content in models.items():
                model = content['model']
                class_names = content['class_names']
                color = content['color']

                # 전처리
                img1, _, _ = resize_with_pad(original, new_shape=(640, 640))
                img = img1[:, :, ::-1].transpose(2, 0, 1)
                img_tensor = torch.from_numpy(np.ascontiguousarray(img)).to(device).float().unsqueeze(0) / 255.0

                with torch.no_grad():
                    pred = model(img_tensor, augment=False, visualize=False)
                    pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.45)[0]

                predictions_for_sorting = []

                if pred is not None and len(pred):
                    pred[:, :4] = scale_boxes((640, 640), pred[:, :4], (h0, w0)).round()
                    annotator = Annotator(annotated, line_width=2, example=str(class_names))

                    for *xyxy, conf, cls in pred:
                        label = class_names[int(cls)]
                        bbox = [int(x) for x in xyxy]
                        predictions_for_sorting.append((label, bbox))

                    ordered_predictions = sort_boxes_order(predictions_for_sorting)

                    pred_dict = {}
                    for label, bbox, order_idx in ordered_predictions:
                        mapped_conf = 0.0
                        for *xyxy, conf_val, cls_idx in pred:
                            bb = [int(x) for x in xyxy]
                            if class_names[int(cls_idx)] == label and bb == bbox:
                                mapped_conf = map_confidence(float(conf_val))
                                break
                        pred_dict[order_idx] = label
                        text = f"[{model_name.split('-')[0]}] {label} {mapped_conf:.2f} ({order_idx})"
                        annotator.box_label(bbox, text, color=color)

                    annotated = annotator.result()

                    # CSV 기록
                    gt_labels = [label for _, label in ground_truth_data.get(model_name, [])]
                    max_idx = max(max(pred_dict.keys(), default=0), len(gt_labels))
                    pred_labels = [pred_dict.get(i, "") for i in range(1, max_idx + 1)]

                    # F1 Score 계산
                    ground_order = ground_truth_data.get(model_name, [])
                    f1_metrics = calculate_order_f1_metrics(ordered_predictions, ground_order)

                    # 기록
                    date_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    image_name = f"{model_name.lower()}_{uuid.uuid4()}.jpg"
                    image_path_str = f"images/{image_name}"
                    Path("images").mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(image_path_str, annotated)

                    # CSV 저장
                    csv_path = Path(prediction_path)
                    csv_path.parent.mkdir(exist_ok=True, parents=True)

                    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['device', model_name.lower()])
                        writer.writerow(['f1score', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', f1_metrics.get('F1_Score', '')])
                        writer.writerow(['date_time', date_time_str])
                        writer.writerow(['image_path', image_path_str])
                        writer.writerow(['ground truth'] + gt_labels)
                        writer.writerow(['prediction'] + pred_labels)

            # 디스플레이
            cv2.imshow("Multi-Model Real-time Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    shortcut_map = {
        'wago':  'WAGO750-1505',
        'bk20s': 'BK20S-T2',
        'bs32c': 'BS32c-6A',
    }

    choice = input("실행할 모델을 입력하세요 (wago / bk20s / bs32c): ").strip().lower()
    if choice not in shortcut_map:
        print(f"잘못된 입력: '{choice}'. wago, bk20s, bs32c 중 하나를 입력하세요.")
        exit(1)

    model_name = shortcut_map[choice]
    print(f">>> {model_name} 모델로 실행합니다...")

    result = check_device(model_name)

    print("\n===== 실행 결과 =====")
    print(f"connect 탐지 횟수   : {result['status']['connect']}")
    print(f"disconnect 탐지 횟수: {result['status']['disconnect']}")
    print(f"F1 Score           : {result['f1_score']}")
    print(f"CSV 경로           : {result['errata_csv']}")
    print(f"이미지 파일        : images/{result['image_name']}")

    realtime_view_with_csv_logging()
    cv2.imshow(f"{model_name} Detection", result['image'])
    print("아무 키나 누르면 종료합니다...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()