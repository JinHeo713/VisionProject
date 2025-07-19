import logging
import pickle
from kafka import KafkaConsumer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZedKafkaConsumer")

def main():
    # KafkaConsumer 생성
    consumer = KafkaConsumer(
        'global_zed',                            # 토픽 이름
        bootstrap_servers='127.0.0.1:29092',     # 브로커 주소
        group_id='zed_consumer_group',           # 그룹 ID
        auto_offset_reset='earliest',            # 처음부터 읽기
        enable_auto_commit=True,                 # 자동 커밋 활성화
        value_deserializer=lambda v: pickle.loads(v),  # 역직렬화
        key_deserializer=lambda k: k.decode('utf-8') if k else None,
    )

    logger.info("KafkaConsumer started. Waiting for messages...")

    try:
        for message in consumer:
            data = message.value

            logger.info(
                f"Received message with key={message.key}, timestamp={data['timestamp']}"
            )

            # 프레임 데이터 구조 예시 출력
            logger.info(f"Color dtype: {data['color_dtype']}, Depth dtype: {data['depth_dtype']}")
            logger.info(f"Color frame size: {len(data['color_data'])} bytes")
            logger.info(f"Depth frame size: {len(data['depth_data'])} bytes")

            # 필요 시 데이터를 복원할 수 있습니다:
            # import numpy as np
            # color_frame = np.frombuffer(data['color_data'], dtype=data['color_dtype']).reshape(HEIGHT, WIDTH, CHANNELS)

    except KeyboardInterrupt:
        logger.info("KafkaConsumer stopped manually.")
    finally:
        consumer.close()
        logger.info("KafkaConsumer closed.")

if __name__ == "__main__":
    main()
