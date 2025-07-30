import pika
import json
from call_mutilmodal import multimodal_analyze_and_save
from save import generate_embeddings_from_csv
import json
import requests
from config import config

RABBITMQ_HOST = config["rabbitmq_host"]
RABBITMQ_USER = config["rabbitmq_user"]
RABBITMQ_PASS = config["rabbitmq_pass"]
QUEUE_NAME = config["queue_name"]

def fetch_user_data_from_api(user_id):
    base_url = config["api_base_url"]
    token = config["api_token"]
    url = f"{base_url}/v1/user/{user_id}/fetch-attrs"
    headers = {
        "Content-Type": "application/json",
        "Cookie": f"token={token}"
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
