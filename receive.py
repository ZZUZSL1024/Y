import pika
import json
from call_mutilmodal import multimodal_analyze_and_save
from save import generate_embeddings_from_csv
import json
import requests
import hashlib
import time

RABBITMQ_HOST = "43.145.37.75"
RABBITMQ_USER = "bixing"
RABBITMQ_PASS = "Bixing@202505"
# QUEUE_NAME = "Test MQ"
## user.modify.attr
## user.change.fragment
QUEUE_NAME = "user.events.queue"

processed_messages = {} 

def is_duplicate(msg_bytes):
    msg_hash = hashlib.md5(msg_bytes).hexdigest()
    now = time.time()
    for k in list(processed_messages):
        if now - processed_messages[k] > 3600:
            del processed_messages[k]
    if msg_hash in processed_messages:
        return True
    processed_messages[msg_hash] = now
    return False

def fetch_user_data_from_api(user_id):
    url = f"http://app.bixing.com.cn/v1/user/{user_id}/fetch-attrs"
    headers = {
        "Content-Type": "application/json",
        "Cookie": "token=1b6a1051a3c645ea9bcbde36435fceef"
    }
    response = requests.post(url, headers=headers, json={})
    if response.status_code == 200:
        res_json = response.json()
        if res_json["code"] == 200:
            return res_json["data"]
        else:
            print(f"[!] API 返回错误：{res_json['message']}")
    else:
        print(f"[!] 请求失败，状态码：{response.status_code}")
    return None

def on_message(ch, method, properties, body):

    print("[x] Received user data from MQ")
    
    data1 = json.loads(body.decode("utf-8"))
    if isinstance(data1, list):
        user_data_list = []
        for item in data1:
            user_id = item.get("userId")
            if user_id:
                user_data = fetch_user_data_from_api(user_id)
                if user_data:
                    user_data_list.append(user_data)
        data = user_data_list
    else:
        user_id = data1.get("userId")
        if user_id:
            data = fetch_user_data_from_api(user_id)
        else:
            print("[!] 没有 userId，跳过处理")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

    if isinstance(data, list):
        for user_json in data:
            multimodal_analyze_and_save(user_json)
    else:
        multimodal_analyze_and_save(data)
    generate_embeddings_from_csv()
    print("[x] 向量文件已更新")
    ch.basic_ack(delivery_tag=method.delivery_tag)
# def on_message(ch, method, properties, body):
#     print("[√] 收到消息，已直接确认，不做处理。")
#     ch.basic_ack(delivery_tag=method.delivery_tag)
#     return

def start_consume():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(RABBITMQ_HOST, 5672, credentials=credentials)
    )
    channel = connection.channel()
    channel.queue_declare(queue='user.events.queue', durable=True)
    channel.queue_bind(exchange='user.behavior.events.exchange',
                      queue='user.events.queue',
                      routing_key='user.modify.attr')
    
    channel.queue_bind(exchange='user.behavior.events.exchange',
                      queue='user.events.queue',
                      routing_key='user.change.fragment')

    channel.basic_qos(prefetch_count=1)
    print(f"[*] Waiting for messages from queue '{QUEUE_NAME}' on {RABBITMQ_HOST} ...")
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message, auto_ack=False)
    channel.start_consuming()

if __name__ == "__main__":
    start_consume()
