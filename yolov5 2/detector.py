# detector.py
import torch
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator
from utils.torch_utils import select_device
import os

class YoloDetector:
    def __init__(self, weights_path, device='cpu', img_size=(320, 320)):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.names = self.model.names
        self.img_size = img_size

    def detect(self, image_in, output_path):
        orig_img = cv2.imread(image_in)
        draw_img = orig_img.copy()

        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, self.img_size)
        img_tensor = resized_img.transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).to(self.device).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = self.model(img_tensor, augment=False, visualize=False)
            preds = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)
            pred = preds[0] if len(preds) > 0 else None

        results = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(resized_img.shape[:2], pred[:, :4], orig_img.shape[:2]).round()
            for *xyxy, conf, cls in pred:
                class_name = self.names[int(cls)]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x = x1
                y = y1
                width = x2 - x1
                height = y2 - y1

                # 색상 지정
                color = (0, 255, 0) if class_name == "connected" else (0, 0, 255)
                label = f"{class_name} {conf:.2f}"

                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                result = {
                    "class_label": class_name,
                    "confidence": round(float(conf), 3),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(width, 2),
                    "height": round(height, 2)
                }
                results.append(result)

        # 저장 경로 생성 및 저장
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, draw_img)

        return {
            "image": output_path,
            "results": results
        }


