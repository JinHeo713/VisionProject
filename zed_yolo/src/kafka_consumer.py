import pickle
from kafka import KafkaConsumer

# Kafka Consumer Configuration
conf = {
    'bootstrap_servers': '127.0.0.1:29092',  # Kafka brokers (Change if needed)
    'group_id': 'my-consumer-group',         # Consumer group ID
    'auto_offset_reset': 'latest'            # Start reading from the latest message
}

# Create Consumer instance
consumer = KafkaConsumer(
    'global_zed',                           # Topic to subscribe to
    **conf                                   # Unpack the config dictionary
)

print(f"Consuming messages from topic: 'global_zed'")

try:
    while True:
        # Poll for messages (timeout=1 second)
        for msg in consumer:
            try:
                # Deserialize message using pickle
                decoded_message = pickle.loads(msg.value)
                print(f"Received message: {decoded_message} (Offset: {msg.offset})")
            except pickle.UnpicklingError as e:
                print(f"Error decoding message with pickle: {e}")
                continue  # Skip any message that fails to decode

except KeyboardInterrupt:
    # Graceful shutdown on Ctrl+C
    print("Consumer interrupted")

finally:
    # Close the consumer to release resources
    consumer.close()
    print("Consumer closed")
