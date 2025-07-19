
# zed_yolo.py
import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import pathlib

# YOLO 경로 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 로컬 모듈 임포트
from camera import ZedCamera
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size, non_max_suppression, scale_boxes,
    xyxy2xywh, increment_path, LOGGER
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

pathlib.WindowsPath = pathlib.PosixPath

def run_detection(
        weights='runs/best.pt',  # 모델 경로
        device='',  # CUDA 장치 (예: 0 또는 '0,1,2,3' 또는 'cpu')
        imgsz=(640, 640),  # 추론 크기 (높이, 너비)
        conf_thres=0.25,  # 신뢰도 임계값
        iou_thres=0.45,  # NMS IoU 임계값
        max_det=1000,  # 이미지당 최대 검출 수
        classes=None,  # 특정 클래스만 필터링: --class 0 또는 --class 0 2 3
        agnostic_nms=False,  # 클래스 구분없는 NMS
        line_thickness=3,  # 바운딩 박스 두께 (픽셀)
        hide_labels=False,  # 라벨 숨기기
        hide_conf=False,  # 신뢰도 점수 숨기기
        view_depth=False,  # 깊이 정보 보기
        save_results=False,  # 결과 저장
        project='runs/detect',  # 결과 저장 디렉토리
        name='exp',  # 실험 이름
):
    """ZED 카메라에서 YOLO 객체 감지 실행"""
    # 모델 로드
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 이미지 크기 확인

    # 모델 워밍업
    model.warmup(imgsz=(1, 3, *imgsz))

    # ZED 카메라 시작
    with ZedCamera() as zed:
        print("ZED 카메라가 시작되었습니다.")

        # 무한 루프에서 카메라 프레임 처리
        while True:
            if not zed.is_frame_available():
                continue

            # 컬러 및 깊이 프레임 가져오기
            color_frame, depth_frame, timestamp = zed.get_color_and_depth_frame()
            # BGRA → BGR (3채널로 변환)
            if color_frame.shape[2] == 4:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)

            # YOLO 추론을 위한 이미지 준비
            img = cv2.resize(color_frame, imgsz)
            img = img.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0  # 0 - 255 -> 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # 배치 차원 추가

            # 추론
            pred = model(img, augment=False, visualize=False)

            # NMS (Non-Maximum Suppression)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # 검출 결과 처리
            det = pred[0].cpu().numpy() if len(pred) and len(pred[0]) else []
            
            
            # 원본 이미지에 검출 결과 그리기
            annotator = Annotator(color_frame, line_width=line_thickness, example=str(names))
            if len(det):
                # 이미지 크기에 맞게 바운딩 박스 조정
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], color_frame.shape).round()

                # 결과 표시
                for *xyxy, conf, cls in det:
                    c = int(cls)  # 정수 클래스
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # 깊이 정보 표시 (선택 사항)
                    # if view_depth:
                    #     x_center = int((xyxy[0] + xyxy[2]) / 2)
                    #     y_center = int((xyxy[1] + xyxy[3]) / 2)
                    #     if 0 <= x_center < depth_frame.shape[1] and 0 <= y_center < depth_frame.shape[0]:
                    #         depth_value = depth_frame[y_center, x_center]
                    #         if depth_value < 100:  # 유효 깊이 값
                    #             cv2.putText(color_frame, f"Depth: {depth_value:.2f}m",
                    #                         (int(xyxy[0]), int(xyxy[1]) - 10),
                    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 검출 결과 표시
            result_img = annotator.result()
            cv2.imshow("ZED + YOLO Detection", result_img)

            # 깊이 정보 시각화 (선택 사항)
            if view_depth:
                depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Depth Map", depth_color)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 종료
    cv2.destroyAllWindows()
    print("종료되었습니다.")


def parse_args():
    parser = argparse.ArgumentParser(description="ZED 카메라에서 YOLO객체 감지 실행")
    parser.add_argument('--weights', type=str, default='runs/BK20S_v3.pt', help='모델 가중치 경로')
    parser.add_argument('--device', default='', help='CUDA 장치 (예: 0 또는 cpu)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='추론 크기 h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='신뢰도 임계값')
    parser.add_argument("--classes", default="1", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 임계값')
    parser.add_argument('--view-depth', action='store_true', help='깊이 정보 표시')
    parser.add_argument('--save-results', action='store_true', help='결과 저장')
    parser.add_argument('--project', default='runs/detect', help='결과 저장 디렉토리')
    parser.add_argument('--name', default='exp', help='실험 이름')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # 확장

    run_detection(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        view_depth=args.view_depth,
        save_results=args.save_results,
        project=args.project,
        name=args.name,
    )