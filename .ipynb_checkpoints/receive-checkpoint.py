# -*- coding: utf-8 -*-
import json
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional
import os
import socket
import threading

import pika
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import config
from .call_mutilmodal import multimodal_analyze_and_save

# ---------- 基础配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("receiver")

RABBITMQ_HOST = config["rabbitmq_host"]
RABBITMQ_USER = config["rabbitmq_user"]
RABBITMQ_PASS = config["rabbitmq_pass"]
QUEUE_NAME    = config["queue_name"]

CLIENT_INFO   = os.environ.get("CLIENT_INFO", socket.gethostname())
PROJECT_NAME  = os.environ.get("PROJECT_NAME", config.get("project_name", "bixing-receive"))

def conn_name(role: str) -> str:
    # 连接名：<项目名>-<角色>@<hostname>-<pid>
    return f"{PROJECT_NAME}-{role}@{CLIENT_INFO}-{os.getpid()}"

# ---------- HTTP 重试会话 ----------
HTTP_TOTAL_RETRIES = int(config.get("http_total_retries", 3))
HTTP_BACKOFF       = float(config.get("http_backoff_factor", 0.5))
HTTP_CONNECT_TO    = float(config.get("http_connect_timeout", 3.0))
HTTP_READ_TO       = float(config.get("http_read_timeout", 10.0))

def make_session():
    sess = requests.Session()
    retry = Retry(
        total=HTTP_TOTAL_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

HTTP = make_session()

# ---------- 简单去重 ----------
processed_messages: Dict[str, float] = {}

def calc_msg_hash(msg_bytes: bytes) -> str:
    return hashlib.md5(msg_bytes).hexdigest()

def is_duplicate(msg_hash: str, window_sec: int = 3600) -> bool:
    now = time.time()
    for k in list(processed_messages):
        if now - processed_messages[k] > window_sec:
            del processed_messages[k]
    return msg_hash in processed_messages

def mark_processed(msg_hash: str):
    processed_messages[msg_hash] = time.time()

# ---------- 监控线程 ----------
def start_queue_monitor(host, user, password, queue, interval=15):
    def _run():
        try:
            creds = pika.PlainCredentials(user, password)
            params = pika.ConnectionParameters(
                host=host, credentials=creds, heartbeat=30,
                client_properties={"connection_name": conn_name("monitor")}
            )
            conn = pika.BlockingConnection(params)
            ch = conn.channel()
            while True:
                try:
                    q = ch.queue_declare(queue=queue, durable=True, passive=True)
                    ready = q.method.message_count
                    log.info("[monitor] queue=%s ready=%d", queue, ready)
                except Exception as e:
                    log.warning("[monitor] query queue depth failed: %s", e)
                time.sleep(interval)
        except Exception as e:
            log.warning("[monitor] init failed: %s", e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ---------- 业务 ----------
def fetch_user_data_from_api(user_id: str) -> Optional[dict]:
    base_url = config["api_base_url"]
    token    = config["api_token"]
    url = f"{base_url}/v1/user/{user_id}/fetch-attrs"
    headers = {"Content-Type": "application/json", "Cookie": f"token={token}"}

    t0 = time.time()
    try:
        resp = HTTP.post(url, headers=headers, json={}, timeout=(HTTP_CONNECT_TO, HTTP_READ_TO))
    except Exception as e:
        log.exception("请求用户属性异常 userId=%s err=%s", user_id, e)
        return None
    dt = (time.time() - t0) * 1000
    if resp.status_code != 200:
        log.error("获取用户属性失败 userId=%s status=%s cost=%.1fms text=%s",
                  user_id, resp.status_code, dt, resp.text[:200])
        return None

    try:
        data = resp.json()
    except Exception:
        log.error("API 返回非 JSON userId=%s cost=%.1fms text[:200]=%s", user_id, dt, resp.text[:200])
        return None

    if data.get("code") == 200 and data.get("data"):
        return data["data"]

    log.error("API 返回业务码非 200 userId=%s cost=%.1fms resp=%s", user_id, dt, data)
    return None

def handle_payload(data):
    if isinstance(data, list):
        for user_json in data:
            multimodal_analyze_and_save(user_json)
    else:
        multimodal_analyze_and_save(data)

def on_message(ch, method, properties, body):
    msg_hash = calc_msg_hash(body)
    if is_duplicate(msg_hash):
        log.info("[dup] 收到重复消息，直接 ack 丢弃")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    try:
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, list):
            user_data_list: List[dict] = []
            for item in payload:
                uid = item.get("userId")
                if not uid:
                    log.warning("批次条目缺少 userId，跳过该条")
                    continue
                ud = fetch_user_data_from_api(str(uid))
                if ud:
                    user_data_list.append(ud)
                else:
                    log.warning("跳过 userId=%s（属性获取失败）", uid)
            if not user_data_list:
                log.warning("批次内无可处理用户，直接 ack")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            data = user_data_list
        else:
            uid = payload.get("userId")
            if not uid:
                log.warning("消息缺少 userId，已 ack 跳过")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            data = fetch_user_data_from_api(str(uid))
            if not data:
                log.warning("未获取到用户属性 userId=%s，已 ack 跳过", uid)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

        t0 = time.time()
        handle_payload(data)
        cost = (time.time() - t0) * 1000
        ch.basic_ack(delivery_tag=method.delivery_tag)
        mark_processed(msg_hash)
        log.info("消息处理完成并已 ack | 业务处理耗时=%.1fms", cost)
    except Exception as e:
        log.exception("处理消息异常，ack 并记录: %s", e)
        ch.basic_ack(delivery_tag=method.delivery_tag)

def start_consume():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=5672,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
        connection_attempts=5,
        retry_delay=2.0,
        client_properties={"connection_name": conn_name("consumer")},
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # ✅ 被动声明：队列必须已存在（含 DLX 等参数），避免 406 不等价
    try:
        channel.queue_declare(queue=QUEUE_NAME, passive=True)
        log.info("队列已存在，被动声明通过：%s", QUEUE_NAME)
    except pika.exceptions.ChannelClosedByBroker as e:
        log.error("被动声明失败（队列不存在或无权限）：%s；err=%s", QUEUE_NAME, e)
        channel = connection.channel()
        raise RuntimeError(
            f"Queue {QUEUE_NAME} not found. Please create it with the correct arguments (e.g. x-dead-letter-exchange)."
        )

    # 绑定单一路由键：gpu.user.update.profile
    RK = "gpu.user.update.profile"
    channel.queue_bind(
        exchange="user.behavior.events.exchange",
        queue=QUEUE_NAME,
        routing_key=RK,
    )

    channel.basic_qos(prefetch_count=1)
    log.info("等待消息中... queue=%s rk=%s host=%s", QUEUE_NAME, RK, RABBITMQ_HOST)
    start_queue_monitor(RABBITMQ_HOST, RABBITMQ_USER, RABBITMQ_PASS, QUEUE_NAME, interval=15)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message, auto_ack=False)
    channel.start_consuming()

if __name__ == "__main__":
    start_consume()
