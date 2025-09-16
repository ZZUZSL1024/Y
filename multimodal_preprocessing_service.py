# -*- coding: utf-8 -*-
"""多模态预处理服务：监听碎片事件并生成结构化描述后回传给后端。"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pika

from .config import config
from .glm_client import GLMClient, extract_json_object

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
GLM_MODEL_NAME = (str(config.get("multimodal_glm_model") or config.get("glm_model_name") or "glm-4.5v")).strip() or "glm-4.5v"
GLM_TIMEOUT = float(config.get("multimodal_glm_timeout", config.get("glm_timeout_sec", 60.0)))
GLM_API_KEY = (
    config.get("zhipuai_api_key")
    or config.get("glm_api_key")
    or os.getenv("ZHIPUAI_API_KEY", "")
)
GLM_BASE_URL = (
    str(config.get("glm_api_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions")).strip()
    or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
)

RABBITMQ_HOST = config["rabbitmq_host"]
RABBITMQ_USER = config["rabbitmq_user"]
RABBITMQ_PASS = config["rabbitmq_pass"]
FRAGMENT_QUEUE_NAME = config.get("fragment_queue_name", "fragment.preprocessing.queue")
FRAGMENT_ROUTING_KEY = config.get("fragment_routing_key", "user.change.fragment")
FRAGMENT_EXCHANGE = config.get("fragment_exchange", "user.behavior.events.exchange")
FRAGMENT_RESULT_EXCHANGE = config.get("fragment_result_exchange", "fragment.preprocessing.exchange")
FRAGMENT_RESULT_ROUTING_KEY = config.get("fragment_result_routing_key", "fragment.preprocessed")

FRAGMENT_ID_FIELD = str(config.get("fragment_id_field", "fragmentId")).strip() or "fragmentId"
FRAGMENT_USER_FIELD = str(config.get("fragment_user_field", "userId")).strip() or "userId"

CLIENT_INFO = os.environ.get("CLIENT_INFO", socket.gethostname())
PROJECT_NAME = os.environ.get("PROJECT_NAME", config.get("project_name", "bixing-preprocessor"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fragment-preprocessor")

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def conn_name(role: str) -> str:
    return f"{PROJECT_NAME}-{role}@{CLIENT_INFO}-{os.getpid()}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if any(sep in text for sep in [",", "、", "|"]):
        return [part.strip() for part in re.split(r"[,、|]", text) if part.strip()]
    return [text]


PROMPT_TEXT = (
    "你是一个专业的多模态理解助手。"\
    "请分析给定的图片或视频关键帧，返回一个 JSON 对象，包含："\
    "description（约 30~60 字的中文描述）和 tags（3~8 个中文标签）。"\
    "标签需涵盖主体、场景、风格或情绪。"
)

MEDIA_URL_FIELDS = ["mediaUrl", "sourceUrl", "url", "coverUrl", "thumbnailUrl"]
MEDIA_TYPE_FIELDS = ["mediaType", "type", "fragmentType", "category"]


class MultimodalPreprocessingService:
    def __init__(self) -> None:
        self.glm_client = GLMClient(
            api_key=GLM_API_KEY,
            model_name=GLM_MODEL_NAME,
            base_url=GLM_BASE_URL,
            timeout=GLM_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # 解析与处理
    # ------------------------------------------------------------------
    def _extract_fragment(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = payload
        if isinstance(payload.get("fragment"), dict):
            data = payload["fragment"]

        fragment_id = str(
            data.get(FRAGMENT_ID_FIELD)
            or data.get("fragmentId")
            or data.get("id")
            or payload.get("fragmentId")
            or payload.get("id")
            or ""
        ).strip()
        if not fragment_id:
            return None

        user_id = str(
            payload.get(FRAGMENT_USER_FIELD)
            or data.get(FRAGMENT_USER_FIELD)
            or payload.get("userId")
            or data.get("userId")
            or payload.get("uid")
            or data.get("uid")
            or ""
        ).strip()

        media_url = None
        for field in MEDIA_URL_FIELDS:
            url = data.get(field) or payload.get(field)
            if isinstance(url, str) and url.strip():
                media_url = url.strip()
                break

        media_type = None
        for field in MEDIA_TYPE_FIELDS:
            value = data.get(field) or payload.get(field)
            if isinstance(value, str) and value.strip():
                media_type = value.strip().lower()
                break

        if not media_url:
            return None

        return {
            "fragment_id": fragment_id,
            "user_id": user_id,
            "media_url": media_url,
            "media_type": media_type or "picture",
        }

    def _build_messages(self, media_url: str, media_type: str) -> List[Dict[str, Any]]:
        description = PROMPT_TEXT
        if media_type and "video" in media_type:
            description += " 如果是视频，请根据画面主要内容做概括。"
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description},
                    {"type": "image_url", "image_url": {"url": media_url}},
                ],
            }
        ]

    def _call_glm(self, fragment_id: str, media_url: str, media_type: str) -> Optional[Dict[str, Any]]:
        messages = self._build_messages(media_url, media_type)
        start = time.time()
        content = self.glm_client.chat(
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        cost = (time.time() - start) * 1000
        if not content:
            log.error("GLM 返回空结果 fragmentId=%s cost=%.1fms", fragment_id, cost)
            return None
        data = extract_json_object(content)
        if not data:
            log.error(
                "GLM 返回无法解析的结果 fragmentId=%s cost=%.1fms content[:200]=%s",
                fragment_id,
                cost,
                content[:200],
            )
            return None

        description = data.get("description") or data.get("caption") or ""
        tags = _ensure_list(data.get("tags") or data.get("keywords"))
        if (not description or not description.strip()) and tags:
            description = "、".join(tags[:5])
        return {
            "description": description.strip(),
            "tags": tags,
            "model_version": GLM_MODEL_NAME,
            "processed_at": _now_iso(),
            "raw_response": data,
        }

    def _publish(self, channel: pika.channel.Channel, message: Dict[str, Any]) -> bool:
        if not FRAGMENT_RESULT_EXCHANGE or not FRAGMENT_RESULT_ROUTING_KEY:
            log.error(
                "未配置 fragment_result_exchange 或 routing_key，无法回传预处理结果 fragmentId=%s",
                message.get("fragmentId"),
            )
            return False
        try:
            channel.basic_publish(
                exchange=FRAGMENT_RESULT_EXCHANGE,
                routing_key=FRAGMENT_RESULT_ROUTING_KEY,
                body=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            log.info(
                "已推送预处理结果 fragmentId=%s exchange=%s rk=%s",
                message.get("fragmentId"),
                FRAGMENT_RESULT_EXCHANGE,
                FRAGMENT_RESULT_ROUTING_KEY,
            )
            return True
        except Exception as exc:
            log.exception("推送预处理结果失败 fragmentId=%s err=%s", message.get("fragmentId"), exc)
            return False

    # ------------------------------------------------------------------
    # 消费逻辑
    # ------------------------------------------------------------------
    def _handle(self, channel, method, properties, body: bytes) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            log.error("解析消息失败 err=%s body=%s", exc, body[:200])
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        items = payload if isinstance(payload, list) else [payload]
        success = True
        for item in items:
            if not isinstance(item, dict):
                log.warning("消息体非字典，跳过：%s", item)
                continue
            info = self._extract_fragment(item)
            if not info:
                log.warning("消息中缺少必要字段，跳过：%s", item)
                continue
            fragment_id = info["fragment_id"]
            media_url = info["media_url"]
            media_type = info["media_type"]
            user_id = info["user_id"]

            features = self._call_glm(fragment_id, media_url, media_type)
            if not features:
                success = False
                continue
            publish_payload = {
                "fragmentId": fragment_id,
                "userId": user_id,
                "mediaUrl": media_url,
                "multimodal_features": features,
            }
            published = self._publish(channel, publish_payload)
            if not published:
                success = False

        channel.basic_ack(delivery_tag=method.delivery_tag)
        if success:
            log.info("消息处理完成 deliveryTag=%s", method.delivery_tag)
        else:
            log.warning("消息处理存在失败的碎片 deliveryTag=%s", method.delivery_tag)

    def start(self) -> None:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=5672,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=300,
            connection_attempts=5,
            retry_delay=2.0,
            client_properties={"connection_name": conn_name("fragment-worker")},
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        # 队列需提前创建好；若不存在则抛异常提醒配置
        try:
            channel.queue_declare(queue=FRAGMENT_QUEUE_NAME, durable=True, passive=True)
        except pika.exceptions.ChannelClosedByBroker:
            channel = connection.channel()
            raise RuntimeError(
                f"Queue {FRAGMENT_QUEUE_NAME} not found. 请在 RabbitMQ 中创建该队列后再启动服务。"
            )

        channel.queue_bind(
            exchange=FRAGMENT_EXCHANGE,
            queue=FRAGMENT_QUEUE_NAME,
            routing_key=FRAGMENT_ROUTING_KEY,
        )
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=FRAGMENT_QUEUE_NAME, on_message_callback=self._handle, auto_ack=False)

        log.info(
            "多模态预处理服务已启动 queue=%s exchange=%s rk=%s",  # noqa: G004
            FRAGMENT_QUEUE_NAME,
            FRAGMENT_EXCHANGE,
            FRAGMENT_ROUTING_KEY,
        )
        channel.start_consuming()


def start_multimodal_preprocessor() -> None:
    service = MultimodalPreprocessingService()
    service.start()


if __name__ == "__main__":
    start_multimodal_preprocessor()
