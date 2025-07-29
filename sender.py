import pika
import json
import os
import argparse

RABBITMQ_HOST = "43.145.37.75"
RABBITMQ_USER = "bixing"
RABBITMQ_PASS = "Bixing@202505"
QUEUE_NAME = "Test MQ"

def send_user_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)
    msg = json.dumps(user_data, ensure_ascii=False)
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(RABBITMQ_HOST, 5672, credentials=credentials)
    )
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=msg.encode("utf-8"),
        properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
    )
    print(f"[x] Sent user data from {json_path} to queue '{QUEUE_NAME}' on {RABBITMQ_HOST}")
    connection.close()

def send_batch_user_data(folder):
    for fname in os.listdir(folder):
        if fname.endswith('.json'):
            send_user_data(os.path.join(folder, fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send user data JSON(s) to RabbitMQ.")
    parser.add_argument("--file", type=str, help="Path to a user JSON file.")
    parser.add_argument("--folder", type=str, help="Path to a folder with user JSON files.")
    args = parser.parse_args()

    if args.file:
        send_user_data(args.file)
    elif args.folder:
        send_batch_user_data(args.folder)
    else:
        print("Please specify --file or --folder.")
