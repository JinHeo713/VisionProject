import torch
import cv2
import numpy as np
import os
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YoloDetector:
    def __init__(self, weights_path, device='cpu', img_size=(320, 320)):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.names = self.model.names
        self.img_size = img_size

        self.sub_model_map = {
            'BK20S-T2': 'runs/train/connect_yolov5s_results/weights/best_BK20.pt',
            'BS32c-6A': 'runs/train/connect_yolov5s_results/weights/best_BS32.pt',
            'WAGO750-1505': 'runs/train/connect_yolov5s_results/weights/best_wago.pt'
        }

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.font_thickness = 2

    def box_overlap(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def draw_label_with_background(self, img, text, topleft, text_color, bg_color):
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        x, y = topleft
        cv2.rectangle(img, (x, y - text_h - 6), (x + text_w + 4, y), bg_color, -1)
        cv2.putText(img, text, (x + 2, y - 2), self.font, self.font_scale, text_color, self.font_thickness, lineType=cv2.LINE_AA)

    def detect(self, image_in, output_path):
        orig_img = cv2.imread(image_in)
        draw_img = orig_img.copy()

        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred1 = self.model(img_tensor, augment=False, visualize=False)
            preds1 = non_max_suppression(pred1, conf_thres=0.5, iou_thres=0.45)
            pred1 = preds1[0] if len(preds1) > 0 else None

        devices_info = []

        if pred1 is not None and len(pred1):
            pred1[:, :4] = scale_boxes(img_tensor.shape[2:], pred1[:, :4], orig_img.shape[:2]).round()
            for *xyxy, conf, cls in pred1:
                cls_name = self.names[int(cls)]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                box1 = [x1, y1, x2, y2]

                color = (255, 0, 0) if cls_name == "WAGO750-1505" else \
                        (0, 255, 255) if cls_name == "BK20S-T2" else \
                        (0, 165, 255)
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
                self.draw_label_with_background(draw_img, label, (x1, y1), text_color=(255, 255, 255), bg_color=color)

                sub_results = []

                if cls_name in self.sub_model_map:
                    sub_model = DetectMultiBackend(self.sub_model_map[cls_name], device=self.device)
                    with torch.no_grad():
                        pred2 = sub_model(img_tensor, augment=False, visualize=False)
                        preds2 = non_max_suppression(pred2, conf_thres=0.5, iou_thres=0.45)
                        pred2 = preds2[0] if len(preds2) > 0 else None

                    if pred2 is not None and len(pred2):
                        pred2[:, :4] = scale_boxes(img_tensor.shape[2:], pred2[:, :4], orig_img.shape[:2]).round()

                        for *xyxy2, conf2, cls2 in pred2:
                            x3, y3, x4, y4 = [int(v) for v in xyxy2]

                            if not self.box_overlap([x3, y3, x4, y4], box1):
                                continue

                            cls_name2 = sub_model.names[int(cls2)]
                            label2 = f"{cls_name2} {conf2:.2f}"
                            color2 = (0, 255, 0) if cls_name2 == "connect" else (0, 0, 255)

                            cv2.rectangle(draw_img, (x3, y3), (x4, y4), color2, 2)
                            self.draw_label_with_background(draw_img, label2, (x3, y3), text_color=(255, 255, 255), bg_color=color2)

                            sub_results.append({
                                "class_label": cls_name2,
                                "confidence": round(float(conf2), 3),
                                "x": round(x3, 2),
                                "y": round(y3, 2),
                                "width": round(x4 - x3, 2),
                                "height": round(y4 - y3, 2)
                            })

                devices_info.append({
                    "device": cls_name,
                    "results": sub_results
                })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, draw_img)

        return {
            "image": output_path,
            "devices": devices_info
        }
