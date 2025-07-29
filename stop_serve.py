import pika
from config import config

RABBITMQ_HOST = config["rabbitmq_host"]
RABBITMQ_USER = config["rabbitmq_user"]
RABBITMQ_PASS = config["rabbitmq_pass"]
QUEUE_NAME = config["queue_name"]

credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(RABBITMQ_HOST, 5672, credentials=credentials)
)
channel = connection.channel()
channel.queue_purge(queue=QUEUE_NAME)
connection.close()
print("队列已清空！")
