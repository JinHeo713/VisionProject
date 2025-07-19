from abc import ABC, abstractclassmethod


class KafkaProducerBase(ABC):
    @abstractclassmethod
    def send_frame(self) -> None:
        ...

    @abstractclassmethod
    def start_stream(self, fps: float = 30.0) -> None:
        ...