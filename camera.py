import time

import cv2

from src.camera.camera_base import Camera, DepthCamera

import logging
# import pyzed.sl as sl
import numpy as np


class GenericCamera(Camera):
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None

    def open(self, params=None):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set timeout to 5 seconds
        start_time = time.time()
        while not self.cap.isOpened():
            if time.time() - start_time > 5:
                raise TimeoutError(f"Failed to connect to camera {self.camera_id} within 5 seconds")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        logging.info(f"Camera {self.camera_id} opened.")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_frame_available(self):
        if self.cap is None:
            return False
        return self.cap.isOpened()

    def get_color_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("Camera frame is not available")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return frame, timestamp


class ZedCamera(DepthCamera):
    def __init__(self, resolution=sl.RESOLUTION.HD720, units=sl.UNIT.METER):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self._zed = sl.Camera()
        self._runtime_params = sl.RuntimeParameters()
        self._left_mat = sl.Mat()
        self._right_mat = sl.Mat()
        self._depth_mat = sl.Mat()
        self._disparity_mat = sl.Mat()
        self._point_cloud_mat = sl.Mat()

        # Initialize ZED camera settings
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = 0.15
        init_params.depth_maximum_distance = 10.0
        init_params.depth_stabilization = True # 고정된 카메라의 경우 False하여 연산성능 향상
        init_params.confidence_threshold = 50
        init_params.textureness_threshold = 50

        # Connect ZED camera
        self.open(init_params)

    def open(self, params=None):
        status = self._zed.open(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera initialization failed: {status}")

    def close(self):
        self._zed.close()
        self._logger.info("ZedCamera closed.")

    def __enter__(self) -> "ZedCamera":
        self._logger.info("ZedCameraProvider Entered")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        # Pass exceptions to the upper level
        return False

    def is_frame_available(self):
        available = (self._zed.grab(self._runtime_params) == sl.ERROR_CODE.SUCCESS)
        return available

    def get_camera_information(self) -> dict:
        camera_info = self._zed.get_camera_information()
        intrinsic = camera_info.camera_calibration_parameters.left_cam
        return {
            "serial_number": camera_info.serial_number,
            "fx": intrinsic.fx,
            "fy": intrinsic.fy,
            "cx": intrinsic.cx,
            "cy": intrinsic.cy,
            "distortion": intrinsic.distortion_coeffs,
        }

    def get_color_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera frame is not available.")
        self._zed.retrieve_image(self._left_mat, sl.VIEW.LEFT)
        color_array = self._left_mat.get_data()
        return color_array

    def get_depth_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera frame is not available.")
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        depth_array = self._depth_mat.get_data()
        return depth_array

    def get_disparity_frame(self) -> np.ndarray:
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera frame is not available.")

        status = self._zed.retrieve_measure(self._disparity_mat, sl.MEASURE.DISPARITY)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera disparity retrieval failed: {status}")

        disparity_array: np.ndarray = self._disparity_mat.get_data()
        return disparity_array

    def get_point_cloud(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera frame is not available.")

        # Get XYZ coord and RGBA color data
        status = self._zed.retrieve_measure(self._point_cloud_mat, sl.MEASURE.XYZRGBA)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera point cloud retrieval failed: {status}")

        point_cloud_array: np.ndarray = self._point_cloud_mat.get_data()
        return point_cloud_array

    def get_color_and_depth_frame(self) -> tuple[np.ndarray, np.ndarray, float]:
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera frame is not available.")

        ts = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        timestamp = ts / 1e9  # nano sec -> sec

        self._zed.retrieve_image(self._left_mat, sl.VIEW.LEFT)
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)

        color_array = self._left_mat.get_data()
        depth_array = self._depth_mat.get_data()

        return color_array, depth_array, timestamp

    def save_snapshot(self, color_path: str, depth_path: str):
        color, depth, _ = self.get_color_and_depth_frame()

        cv2.imwrite(color_path, color)
        self._logger.info(f"Saved snapshot to {color_path} and {depth_path}.")

        depth_normalized = cv2.normalize(
            depth, None,
            alpha=0, beta=65535,
            norm_type=cv2.NORM_MINMAX
        )
        depth_uint16 = depth_normalized.astype(np.uint16)
        cv2.imwrite(depth_path, depth_uint16)
        self._logger.info(f"Saved snapshot to {color_path} and {depth_path}.")

    def record_video(self, color_video_path: str, depth_video_path: str, fps: float = 30.0) -> None:
        # Get resolution from the first frame
        color, depth, _ = self.get_color_and_depth_frame()
        height, width = color.shape[:2]

        # VideoWriter setup: BGR for color, grayscale(single-channel) for depth
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vw_color = cv2.VideoWriter(color_video_path, fourcc, fps, (width, height))
        vw_depth = cv2.VideoWriter(depth_video_path, fourcc, fps, (width, height), isColor=False)

        self._logger.info(f"Started recording to {color_video_path} and {depth_video_path}")

        try:
            while True:
                # Check frame availability and get frames
                color, depth, _ = self.get_color_and_depth_frame()

                # Record color frame
                vw_color.write(color)

                # Convert depth frame to 8-bit and record
                # (Scale based on max value since original could be float32)
                depth_8bit = cv2.convertScaleAbs(
                    depth, alpha=255.0 / (np.max(depth) if np.max(depth) > 0 else 1.0)
                )
                vw_depth.write(depth_8bit)

        except KeyboardInterrupt:
            self._logger.info("Recording stopped by user.")
        finally:
            vw_color.release()
            vw_depth.release()
            self._logger.info("VideoWriter resources released.")
