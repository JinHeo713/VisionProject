import logging
import time
import pickle
import threading
import queue
from typing import Any
from kafka import KafkaProducer
from kafka_base import KafkaProducerBase
from camera import ZedCamera


class ZedKafkaProducer(KafkaProducerBase):
    def __init__(self, camera: ZedCamera, topic: str, bootstrap_server=None):
        super().__init__()

        if bootstrap_server is None:
            bootstrap_server = ["127.0.0.1:29092"]

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        # Zed cam Dependency injection 
        self._camera = camera
        
        # kafka producer topic
        self._topic = topic

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_server,
            key_serializer=lambda k: k.encode("utf-8"),
            value_serializer=lambda v: pickle.dumps(v),
            max_request_size=33_554_432,
            buffer_memory=67_108_864
        )

        # Queue for inter-thread communication
        self._frame_queue = queue.Queue()

    def __enter__(self) -> "ZedKafkaProducer":
        # entering camera context
        self._camera.__enter__()
        self._logger.info("ZedKafkaProducer Entered")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # exit kafka Producer
        try:
            self._producer.close()
            self._logger.info("ZedKafkaProducer closed")
        except Exception as e:
            self._logger.error(f"ZedKafkaProducer close failed: {e}")
            self._camera.__exit__(exc_type, exc_val, exc_tb)
            return False

    def capture_frame(self):
        """Threaded function to capture frames."""
        try:
            while True:
                color_frame, depth_frame, timestamp = self._camera.get_color_and_depth_frame()

                # message
                message: dict[str, Any] = {
                    "timestamp": timestamp,
                    "color_frame": color_frame,
                    "depth_frame": depth_frame,
                    "color_dtype": color_frame.dtype,
                    "depth_dtype": depth_frame.dtype,
                    "color_data": color_frame.tobytes(),
                    "depth_data": depth_frame.tobytes(),
                }

                # Put the frame into the queue for the sending thread
                self._frame_queue.put(message)
                time.sleep(0.03)  # 30 FPS

        except KeyboardInterrupt:
            self._logger.info("Frame capture stopped by user.")
        except Exception as e:
            self._logger.error(f"Error capturing frame: {e}")

    def send_frame(self):
        """Threaded function to send frames to Kafka."""
        try:
            while True:
                # Wait for a frame from the capture thread
                message = self._frame_queue.get()

                # Send to Kafka broker
                key_str = str(message['timestamp'])
                self._producer.send(self._topic, key=key_str, value=message)
                self._producer.flush()
                self._logger.info(f"Message sent to topic {self._topic} with key {key_str}")

                self._frame_queue.task_done()

        except KeyboardInterrupt:
            self._logger.info("Message sending stopped by user.")
        except Exception as e:
            self._logger.error(f"Error sending message: {e}")

    def start_stream(self, fps: float = 30.0) -> None:
        self._logger.info(f"Sending stream to topic {self._topic} at {fps} FPS.")
        
        # Create threads for capturing frames and sending frames
        capture_thread = threading.Thread(target=self.capture_frame, daemon=True)
        send_thread = threading.Thread(target=self.send_frame, daemon=True)

        # Start both threads
        capture_thread.start()
        send_thread.start()

        # Wait for the threads to finish (in practice, we may want them to run indefinitely)
        try:
            capture_thread.join()
            send_thread.join()
        except KeyboardInterrupt:
            self._logger.info("Stream stopped by user.")
        finally:
            self._producer.flush()
            self._logger.info("Stream finished")
            self._camera.close()
            self._logger.info("Resources released")


if __name__ == "__main__":
    zcam = ZedCamera()
    z_producer = ZedKafkaProducer(zcam, "global_zed")

    # Start streaming in separate threads
    z_producer.start_stream(fps=30.0)
