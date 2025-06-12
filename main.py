#main.py
# from fastapi import FastAPI
# from detector import YoloDetector
# import json

# app = FastAPI()

# detector = YoloDetector(weights_path="runs/train/connect_yolov5s_results2/weights/best_window.pt")

# @app.post("/vision/qc/check")
# def detect():
#     img_path = "dataset/input_image/25cm_30.jpg"
    
#     result = detector.detect(img_path)
#     return {"result": result}



