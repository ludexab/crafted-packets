import pika
import time
import json
import random

amqp_queue = 'sensor_data'

def publish_sensor_data():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=amqp_queue)

    while True:
        sensor_data = {
            'sensor_id': 'sensor_1',
            'temperature': random.uniform(20.0, 30.0),
            'humidity': random.uniform(30.0, 50.0),
            'timestamp': int(time.time())
        }
        message = json.dumps(sensor_data)
        channel.basic_publish(exchange='',
                              routing_key=amqp_queue,
                              body=message)
        print(f"Published: {message}")
        time.sleep(0.1)

    connection.close()

if __name__ == "__main__":
    publish_sensor_data()


selected_columns = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean',
    'Bwd Packet Length Mean', 'Packet Length Mean', 'Packet Length Std',
    'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Label'
]
