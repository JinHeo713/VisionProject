from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import base64
import cv2
import uuid

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from detector import YoloDetector

app = FastAPI()

# 요청 및 응답 모델 정의
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

# 루트 엔드포인트
@app.get("/health")
async def read_root():
    return {"message": "OK"}


@app.get("/vision/status")
async def read_root():
    return {"message": "OK"}


# 쿼리 엔드포인트
@app.get("/vision/qc/check")
async def vision_qc_check():
    # YOLO 모델 불러오기 및 객체 생성
    detector = YoloDetector(weights_path="../runs/train/connect_yolov5s_results2/weights/best_window.pt")

    cap = cv2.VideoCapture(0)  # 0번 카메라 (기본 카메라)
    ret, frame = cap.read()
    cap.release()

    image_path = f"camera_input_{uuid.uuid4()}.jpg"
    cv2.imwrite(image_path, frame)

    # YOLO 예측 및 결과 반환
    # result = {
    #     "class_label": String  # (connected / disconnected)
    #     "confidence": double,
    #     "x": double,
    #     "y": double,
    #     "width": double,
    #     "height": double
    # }
    result = detector.detect(image_path, image_path)
    vision_qc_result = {
        "result": result
    }

    image_path = result["image"]
    return vision_qc_result

# 이미지 반환
@app.get("/vision/qc/check/image/{image_path}")
async def vision_qc_check_image(image_path: str):
    return FileResponse("dataset/input_image/" + image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)