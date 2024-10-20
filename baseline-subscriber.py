import pika
import json

amqp_queue = 'sensor_data'

def callback(ch, method, properties, body):
    sensor_data = json.loads(body)
    print(f"Received: {sensor_data}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_sensor_data():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=amqp_queue)

    channel.basic_consume(queue=amqp_queue,
                           on_message_callback=callback,
                             auto_ack=False)
    
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    consume_sensor_data()
