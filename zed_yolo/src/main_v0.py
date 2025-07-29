import torch
import cv2
import numpy as np
import time
from collections import Counter
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device
from camera import ZedCamera
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import base64
import cv2
import uuid
import APItest_v4

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
app = FastAPI()

# 요청 및 응답 모델 정의
class Item(BaseModel):
    name: str
    description: str
    price: float
    tax: float

# 루트 엔드포인트
@app.get("/health")
async def read_root():
    return {"message": "OK"}

@app.get("/vision/status")
async def read_root():
    return {"message": "OK"}

# ───── 설정: 모델명 및 경로 ──────────────────────────────────────────
@app.get("/vision/qc/check/wago")
async def check_wago():
    result = APItest_v4.check_device('WAGO750-1505')
    if result["status"] == "failed":
        return result

    save_folder = 'images'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, result["image_name"])

    cv2.imwrite(save_path, result["image"])

    # Remove the image from result before returning if you want just JSON
    return {
        "status": result["status"],
        "confidence": result["confidence"],
        "image_name": result["image_name"],
        "f1_metrics": result["f1_score"]
    }

@app.get("/vision/qc/check/bk20")
async def check_bk20():
    result = APItest_v4.check_device('BK20S-T2')
    if result["status"] == "failed":
        return result

    save_folder = 'images'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, result["image_name"])

    cv2.imwrite(save_path, result["image"])

    return {
        "status": result["status"],
        "confidence": result["confidence"],
        "image_name": result["image_name"],
        "f1_metrics": result["f1_score"]
    }

@app.get("/vision/qc/check/bs32")
async def check_bs32():
    result = APItest_v4.check_device('BS32c-6A')
    if result["status"] == "failed":
        return result

    save_folder = 'images'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, result["image_name"])

    cv2.imwrite(save_path, result["image"])

    return {
        "status": result["status"],
        "confidence": result["confidence"],
        "image_name": result["image_name"],
        "f1_metrics": result["f1_score"]
    }

# 이미지 반환
@app.get("/vision/qc/check/image/{image_path}")
async def vision_qc_check_image(image_path: str):
    return FileResponse("images/" + image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)