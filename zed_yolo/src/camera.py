import cv2
import time
import logging
import pyzed.sl as sl

from typing import Tuple
from camera_base import DepthCamera

class ZedCamera(DepthCamera):
    def __init__(self, resolution=sl.RESOLUTION.HD720, units=sl.UNIT.METER):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self._zed = sl.Camera()
        self._runtime_parmas = sl.RuntimeParameters()
        self._left_mat = sl.Mat()
        self._depth_mat = sl.Mat()
        self._disparity_mat = sl.Mat()
        self._pointcloud_mat = sl.Mat()

        # Initialize ZED camera settings
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = units
        init_params.depth_minimum_distance = 0.15
        init_params.depth_maximum_distance = 2.0
        init_params.depth_stabilization = 1  # if camera fixed -> False

        self.open(init_params)
        
    def open(self, params=None):
        status = self._zed.open(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera initialization failed: {status}")
        
    def close(self):
        self._zed.close()
        self._logger.info("ZED camera closed")

    def __enter__(self) -> "ZedCamera":
        self._logger.info("ZedCamera Entered")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # Pass exceptions to the upper level
    
    def is_frame_available(self):
        available = (self._zed.grab(self._runtime_parmas) == sl.ERROR_CODE.SUCCESS)
        return available
    
    def get_camera_information(self) -> dict:
        camera_info = self._zed.get_camera_information()
        intrinsic = camera_info.camera_configuration.calibration_parameters.left_cam
        return {
            "serial_number": camera_info.serial_number,
            "fx": intrinsic.fx,
            "fy": intrinsic.fy,
            "cx": intrinsic.cx,
            "cy": intrinsic.cy,
        }
    
    def get_color_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera is not available")
        self._zed.retrieve_image(self._left_mat, sl.VIEW.LEFT)
        color_array = self._left_mat.get_data()
        return color_array
    
    def get_depth_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera is not available")
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)
        depth_array = self._depth_mat.get_data()
        return depth_array
    
    def get_disparity_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera is not available")
        
        status = self._zed.retrieve_measure(self._disparity_mat, sl.MEASURE.DISPARITY)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED Camera disparity retrieval failed: {status}")
        
        disparity_array = self._disparity_mat.get_data()
        return disparity_array
    
    def get_pointcloud(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera is not available")
        
        # Get XYZ coord and RGBA color data
        status = self._zed.retrieve_measure(self._pointcloud_mat, sl.MEASURE.XYZRGBA)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED Camera point cloud retrieval failed: {status}")
        
        pointcloud_array = self._pointcloud_mat.get_data()
        return pointcloud_array
    
    def get_color_and_depth_frame(self):
        if not self.is_frame_available():
            raise RuntimeError("ZED Camera is not available")
        
        ts = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        timestamp = ts / 1e9

        self._zed.retrieve_image(self._left_mat, sl.VIEW.LEFT)
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH)

        color_array = self._left_mat.get_data()
        depth_array = self._depth_mat.get_data()
        return color_array, depth_array, timestamp
