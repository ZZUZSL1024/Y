import pika

RABBITMQ_HOST = "43.145.37.75"
RABBITMQ_USER = "bixing"
RABBITMQ_PASS = "Bixing@202505"
QUEUE_NAME = "user.events.queue"

credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(RABBITMQ_HOST, 5672, credentials=credentials)
)
channel = connection.channel()
channel.queue_purge(queue=QUEUE_NAME)
connection.close()
print("队列已清空！")
