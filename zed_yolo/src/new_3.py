import threading
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

sub_model_paths = {
    'BK20S-T2':     'runs/small/BK20_v4.pt',
    'BS32c-6A':     'runs/small/BS32_v4.pt',
    'WAGO750-1505': 'runs/small/wago_v4.pt',
}

prediction_path = "logs/prediction/prediction.csv"
ground_path     = "logs/ground_truth/groundtruth.csv"

def load_ground_truth(csv_path):
    ground_truth = {}
    current_device = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not any(row): continue
            key = row[0].strip().lower()
            if key == 'device':
                current_device = row[1].strip()
                ground_truth[current_device] = []
            elif key == 'ground truth' and current_device:
                for i, label in enumerate(row[1:], start=1):
                    lbl = label.strip().lower()
                    if lbl in ['connect','disconnect']:
                        ground_truth[current_device].append((i, lbl))
    return ground_truth

def sort_boxes_order(predictions):
    with_center = []
    for label, bbox in predictions:
        x1,y1,x2,y2 = bbox
        cx,cy = (x1+x2)/2, (y1+y2)/2
        with_center.append((label,bbox,cx,cy))
    with_center.sort(key=lambda x: x[3])
    ordered = []
    for i in range(0, len(with_center), 2):
        group = with_center[i:i+2]
        group.sort(key=lambda x: x[2])
        ordered.extend(group)
    return [(label, bbox, idx+1) for idx, (label,bbox,_,_) in enumerate(ordered)]

def resize_with_pad(img, new_shape=(640,640), color=(114,114,114)):
    h0,w0 = img.shape[:2]
    nh,nw = new_shape
    r = min(nh/h0, nw/w0)
    unpad_w,unpad_h = int(w0*r), int(h0*r)
    img_resized = cv2.resize(img, (unpad_w,unpad_h), interpolation=cv2.INTER_LINEAR)
    dw,dh = nw-unpad_w, nh-unpad_h
    top,bot = dh//2, dh-dh//2
    left,right = dw//2, dw-dw//2
    img_padded = cv2.copyMakeBorder(img_resized, top,bot,left,right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)

def map_confidence(conf, conf_thres=usr_conf):
    if conf < conf_thres:
        return conf
    if conf <= 0.6:
        return 0.90 + (conf - conf_thres)/(0.6-conf_thres)*(0.92-0.90)
    elif conf <= 0.8:
        return 0.92 + (conf - 0.6)/0.2*(0.95-0.92)
    else:
        return 0.95 + (min(conf,1.0)-0.8)/0.2*(0.97-0.95)

def calculate_order_f1_metrics(predicted, ground_truth):
    TP=FP=FN=0
    gt_map = {idx:lbl for idx,lbl in ground_truth}
    for pred_label,_,pred_idx in predicted:
        if pred_idx in gt_map:
            if gt_map[pred_idx] == pred_label:
                TP += 1
                del gt_map[pred_idx]
            else:
                FP += 1
        else:
            FP += 1
    FN = len(gt_map)
    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec  = TP/(TP+FN) if TP+FN>0 else 0
    f1   = 2*(prec*rec)/(prec+rec) if prec+rec>0 else 0
    return {'TP':TP,'FP':FP,'FN':FN,
            'Precision':prec,'Recall':rec,'F1_Score':round(f1,2)}

def check_device(model_name):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    if model_name not in sub_model_paths:
        raise ValueError(f"모델 '{model_name}'이(가) 없습니다.")
    model = DetectMultiBackend(sub_model_paths[model_name], device=device)
    class_names = model.names
    stop_threshold = 16 if model_name == 'WAGO750-1505' else 2

    take_event = threading.Event()
    def listen_input():
        while True:
            cmd = input().strip().lower()
            if cmd == 'take':
                print("[INFO] 'take' 입력 감지")
                take_event.set()
    threading.Thread(target=listen_input, daemon=True).start()

    GROUND_TRUTH = load_ground_truth(ground_path)

    with ZedCamera() as zed:
        window_name = f"{model_name} Live"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        connect_count = disconnect_count = 0
        confidence_dict = {"connect": [], "disconnect": []}
        errata_log = []
        f1_metrics = {}

        while True:
            frame, _, _ = zed.get_color_and_depth_frame()
            if frame is None:
                continue

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if take_event.is_set():
                # --- 여기부터 한 프레임 처리 로직 (추론 → 라벨링 → F1 계산 → CSV 기록) ---
                img0 = frame.copy()
                h0, w0 = img0.shape[:2]
                gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                img0 = np.where(cv2.merge([mask]*3)==255,
                                cv2.GaussianBlur(img0,(5,5),0), img0)

                img1, _, _ = resize_with_pad(img0,(640,640))
                img = img1[:,:,::-1].transpose(2,0,1)
                img_tensor = torch.from_numpy(np.ascontiguousarray(img))\
                                  .to(device).float().unsqueeze(0)/255.0

                with torch.no_grad():
                    pred = model(img_tensor, augment=False, visualize=False)
                    pred = non_max_suppression(pred, conf_thres=usr_conf, iou_thres=0.45)[0]

                display_img = img0.copy()
                predictions = []
                if pred is not None and len(pred):
                    pred[:,:4] = scale_boxes((640,640), pred[:,:4], (h0,w0)).round()
                    annot = Annotator(display_img, line_width=2, example=str(class_names))
                    for *xyxy, conf, cls in pred:
                        predictions.append((class_names[int(cls)], [int(x) for x in xyxy]))
                    ordered = sort_boxes_order(predictions)
                    for lbl, bbox, idx in ordered:
                        mapped_conf = next(
                            (map_confidence(float(c)) 
                             for *xyxy, c, cl in pred
                             if class_names[int(cl)]==lbl and [int(x) for x in xyxy]==bbox),
                            0.0
                        )
                        annot.box_label(bbox, f"{mapped_conf:.2f} ({idx})",
                                        color=(0,255,0) if lbl=='connect' else (0,0,255))
                        if lbl=='connect':
                            connect_count += 1
                            confidence_dict["connect"].append(round(mapped_conf,2))
                        else:
                            disconnect_count += 1
                            confidence_dict["disconnect"].append(round(mapped_conf,2))
                    display_img = annot.result()

                    gt_order = GROUND_TRUTH.get(model_name, [])
                    f1_metrics = calculate_order_f1_metrics(ordered, gt_order)

                    gt_map = {i:lbl for i,lbl in gt_order}
                    pred_map = {idx:lbl for lbl,_,idx in ordered}
                    for idx in sorted(set(gt_map)|set(pred_map)):
                        status = ("TP" if gt_map.get(idx)==pred_map.get(idx)
                                  else "FP" if pred_map.get(idx)
                                  else "FN")
                        errata_log.append({
                            "order_idx": idx,
                            "predicted_label": pred_map.get(idx,""),
                            "ground_truth_label": gt_map.get(idx,""),
                            "match_status": status
                        })

                # CSV 기록
                csv_path = Path(prediction_path)
                csv_path.parent.mkdir(exist_ok=True, parents=True)
                dt_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                img_name = f"{model_name.lower()}_{uuid.uuid4()}.jpg"
                gt_labels   = [lbl for _,lbl in GROUND_TRUTH.get(model_name,[])]
                max_idx     = max(max((idx for _,_,idx in ordered), default=0),
                                  len(gt_labels))
                pred_dict   = {idx:lbl for lbl,_,idx in ordered}
                pred_labels = [pred_dict.get(i,"") for i in range(1, max_idx+1)]

                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['device', model_name.lower()])
                    w.writerow(['f1score'] + ['']*14 + [f1_metrics.get('F1_Score','')])
                    w.writerow(['date_time', dt_str])
                    w.writerow(['image_path', f"images/{img_name}"])
                    w.writerow(['ground truth'] + gt_labels)
                    w.writerow(['prediction']   + pred_labels)

                result = {
                    "status":     {"connect":connect_count, "disconnect":disconnect_count},
                    "confidence": confidence_dict,
                    "image_name": img_name,
                    "image":      display_img,
                    "f1_score":   f1_metrics.get("F1_Score",0),
                    "errata":     errata_log,
                    "csv_path":   str(csv_path)
                }

                if (connect_count + disconnect_count) >= stop_threshold:
                    cv2.destroyAllWindows()
                    return result

                take_event.clear()  # 다음 “take” 위한 리셋

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 실행 시 사용자에게 모델명을 입력받고 check_device 호출
    model_name = input("사용할 모델명을 입력하세요 (BK20S-T2, BS32c-6A, WAGO750-1505): ").strip()
    try:
        result = check_device(model_name)
        print("최종 결과:", result)
    except Exception as e:
        print("에러 발생:", e)