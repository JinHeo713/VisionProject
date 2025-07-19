from abc import ABC, abstractclassmethod
from typing import Any, Tuple
import numpy as np
import logging


class DepthCamera(ABC):
    @abstractclassmethod
    def open(self, params: Any = None) -> None:
        logging.info(f"{self.__class__.__name__}")

    @abstractclassmethod
    def close(self) -> None:
        logging.info(f"{self.__class__.__name__} resource closed")

    @abstractclassmethod
    def get_camera_information(self) -> dict:
        ...

    @abstractclassmethod
    def get_disparity_frame(self) -> np.ndarray:
        ...

    @abstractclassmethod
    def get_color_frame(self) -> np.ndarray:
        ...

    @abstractclassmethod
    def get_depth_frame(self) -> np.ndarray:
        ...

    @abstractclassmethod
    def get_color_and_depth_frame(self) -> Tuple[np.ndarray, np.ndarray, float]:
        ...

    @abstractclassmethod
    def get_pointcloud(self) -> np.ndarray:
        ...    