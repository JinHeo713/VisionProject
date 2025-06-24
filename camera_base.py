from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import logging

from typing import Tuple


class Camera(ABC):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def open(self, params: Any = None) -> None:
        """Opens camera connection with given parameters.
        
        Args:
            params: Optional camera-specific initialization parameters
            
        Returns:
            None
        """
        logging.info(f"{self.__class__.__name__} resource opened.")

    @abstractmethod
    def close(self) -> None:
        """Releases camera resources and closes connection.
        
        Returns:
            None
        """
        logging.info(f"{self.__class__.__name__} resource closed.")

    @abstractmethod
    def is_frame_available(self) -> bool:
        """Checks if the next frame is ready to be grabbed.
        
        Returns:
            bool: True if the frame is ready, False otherwise
        """
        ...

    @abstractmethod
    def get_color_frame(self) -> np.ndarray:
        """Gets color frame data from the camera.
        
        Returns:
            numpy.ndarray: Array containing the color frame data
            
        Raises:
            RuntimeError: If frame is not ready
        """
        ...


class DepthCamera(ABC):
    @abstractmethod
    def open(self, params: Any = None) -> None:
        """Opens depth camera connection with given parameters.
        
        Args:
            params: Optional camera-specific initialization parameters
            
        Returns:
            None
        """
        logging.info(f"{self.__class__.__name__} resource opened.")

    @abstractmethod
    def close(self) -> None:
        """Releases camera resources and closes connection.
        
        Returns:
            None
        """
        logging.info(f"{self.__class__.__name__} resource closed.")

    @abstractmethod
    def is_frame_available(self) -> bool:
        """Checks if the next frame is ready to be grabbed.
        
        Returns:
            bool: True if the frame is ready, False otherwise
        """
        ...
    
    @abstractmethod
    def get_camera_information(self) -> dict:
        ...
    
    @abstractmethod
    def get_disparity_frame(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_color_frame(self) -> np.ndarray:
        """Gets color frame data from the camera.
        
        Returns:
            numpy.ndarray: Array containing the color frame data
            
        Raises:
            RuntimeError: If frame is not ready
        """
        ...

    @abstractmethod
    def get_depth_frame(self) -> np.ndarray:
        """Gets depth frame data from the camera.
        
        Returns:
            numpy.ndarray: Array containing depth measurement values in configured units
            
        Raises:
            RuntimeError: If frame is not ready
        """
        ...

    @abstractmethod
    def get_color_and_depth_frame(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns synchronized color and depth frames with corresponding timestamp 
        from a single grab() operation.

        Returns:
            color_frame (np.ndarray): BGR color image
            depth_frame (np.ndarray): Depth map (in configured units)
            timestamp (float): Unix timestamp (seconds) at grab time
        Raises:
            RuntimeError: If frame preparation fails
        """
        ...
    
    @abstractmethod
    def get_point_cloud(self) -> np.ndarray:
        ...
    
    @abstractmethod
    def save_snapshot(self, color_path: str, depth_path: str):
        ...
    
    @abstractmethod
    def record_video(
            self,
            color_video_path: str,
            depth_video_path: str,
            fps: float = 30.0
    ) -> None:
        """Records synchronized color and depth video streams to files.

        Args:
            color_video_path: Output path for the color video file
            depth_video_path: Output path for the depth video file  
            fps: Frame rate for recorded video (default: 30.0)

        Returns:
            None

        Raises:
            RuntimeError: If video recording fails
            ValueError: If invalid paths or fps are provided
        """
        ...
