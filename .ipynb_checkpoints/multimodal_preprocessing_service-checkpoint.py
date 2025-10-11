# -*- coding: utf-8 -*-
"""多模态预处理服务：监听碎片事件并生成结构化描述后回传给后端。"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
FRAGMENT_RESULT_EXCHANGE = config.get(
    "fragment_result_exchange", "ai.processing.results.exchange"
)
FRAGMENT_RESULT_ROUTING_KEY = config.get(
    "fragment_result_routing_key", "fragment.preprocessing.completed"
)

RESULT_STATUS_SUCCESS = "success"
RESULT_STATUS_FAILURE = "failure"
DEFAULT_ERROR_CODE = "PROCESSING_ERROR"

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

# ====== 新增：两套提示词 ======
PROMPT_IMAGE = r"""
你是一个专业的图像分析引擎，任务是像法证调查员一样，客观、详尽、不带任何主观推断地描述眼前这幅图像。你的描述将被后续的文本分析模型使用，因此必须包含尽可能丰富的视觉细节。

请严格按照以下JSON格式返回你的分析结果，确保输出是一个完整的、格式正确的JSON对象，只包含 "description" 和 "tags" 两个键。

{
  "description": "（在这里生成一段详尽的、多句话的描述。请依次描述以下几个方面：
    1. **核心主体**: 图片的焦点是什么？如果是人，请描述他们的外貌、衣着、姿态、表情和正在做的动作。如果是物体，请描述它的材质、形状、颜色和状态。
    2. **环境背景**: 场景发生在哪里？是室内还是室外？描述背景中的关键元素，例如建筑风格、自然景观（山川、树木、天空）、城市街景等。
    3. **构图与光线**: 描述画面的构图（例如，主体在中心还是偏离中心，前景、中景、背景分别有什么）。描述光线情况（例如，是明亮的日光、柔和的室内灯光还是昏暗的黄昏光线）和整体色调（例如，是暖色调、冷色调还是黑白色）。
    4. **辅助元素**: 画面中还有哪些值得注意的物体或细节？请一一列出并简要描述，它们可以为理解场景提供更多背景信息。）",
  "tags": [
    "（根据图片中可直接观察到的客观元素，生成一个关键词列表。标签应为名词或动名词，例如：'笔记本电脑', '咖啡馆', '人物侧影', '城市夜景', '拉布拉多犬', '海边日落', '徒步旅行'）"
  ]
}
""".strip()

PROMPT_VIDEO = r"""
你是一个专业的视频画面分析引擎。现在将收到按时间顺序抽取的**多张关键帧**（仅视觉，无音频与真实时间戳）。请**综合所有帧**进行客观描述，避免编造时间、音频、因果或主观动机。
请严格按以下 JSON 返回，且只包含 "description" 和 "tags" 两个键。

{
  "description": "请按结构化方式书写：A) 全局概括（视频主要场景与主题）；B) 时间序列要点（按帧出现顺序概括画面变化与动作推进，可用“帧1…帧N”的方式）；C) 场景/镜头变化（地点/光线/构图显著变化）；D) 核心主体与交互（谁在做什么、物体如何被操作——仅基于可见画面）；E) 视觉风格（色调、光线、运动模糊/景深等）；F) 结尾画面状态（最后一帧可见结果）。全程保持客观与克制。",
  "tags": [
    "稳定元素与关键动作的名词/动名词标签，如：'室内办公','会议桌','投影屏幕','人群走动','车辆驶过','海边黄昏','镜头切换','特写','俯拍','低照度'"
  ]
}
""".strip()

# ---------------------------------------------------------------------------
# 可调视频抽帧参数（本地抽帧 -> 多图送入 GLM）
# ---------------------------------------------------------------------------
VIDEO_FRAME_COUNT = int(config.get("video_frame_count", 5))             # 最大抽帧数
VIDEO_MAX_PIXELS = int(config.get("video_max_pixels", 1280 * 720))      # 每帧最大像素（约 720p）
VIDEO_EXTS = set((config.get("video_exts") or ".mp4,.mov,.m4v,.webm,.avi,.mkv").lower().split(","))
MAX_TOTAL_IMAGE_BYTES = int(config.get("video_max_total_image_bytes", 6 * 1024 * 1024))  # 所有帧图总字节上限（base64前原始文件大小估算）

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

def _is_probably_video(media_type: Optional[str], media_url: str) -> bool:
    if media_type and "video" in media_type.lower():
        return True
    try:
        ext = (os.path.splitext(urlparse(media_url).path)[1] or "").lower()
        return ext in VIDEO_EXTS
    except Exception:
        return False

def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _sample_video_frames_to_files(media_url: str, num_frames: int = VIDEO_FRAME_COUNT) -> List[str]:
    """
    使用 ffmpeg 从远程视频均匀抽取若干帧到临时目录，返回帧图文件列表。
    策略：使用 fps=1 降采样并限制分辨率，随后截取前 num_frames 张。
    """
    if not _has_ffmpeg():
        return []

    tmpdir = tempfile.mkdtemp(prefix="vidframes_")
    out_pattern = os.path.join(tmpdir, "frame_%03d.jpg")

    # 缩放到不超过 VIDEO_MAX_PIXELS 的分辨率，并以 1 fps 抽帧
    vf = (
        f"scale='if(gt(iw*ih,{VIDEO_MAX_PIXELS}),iw*sqrt({VIDEO_MAX_PIXELS}/(iw*ih)),iw)':"
        f"'if(gt(iw*ih,{VIDEO_MAX_PIXELS}),ih*sqrt({VIDEO_MAX_PIXELS}/(iw*ih)),ih)',fps=1"
    )
    cmd = ["ffmpeg", "-y", "-i", media_url, "-vf", vf, "-vsync", "vfr", out_pattern]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except Exception as e:
        log.warning("ffmpeg 抽帧失败：%s", e)
        return []

    frames = sorted(
        [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".jpg")]
    )
    if not frames:
        return []

    return frames[: max(1, num_frames)]

def _file_to_data_url(fp: str) -> str:
    with open(fp, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", len(raw)



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

        trace_id = str(
            payload.get("trace_id")
            or payload.get("traceId")
            or data.get("trace_id")
            or data.get("traceId")
            or ""
        ).strip()

        request_metadata: Dict[str, Any] = {}
        for candidate in (payload.get("metadata"), data.get("metadata")):
            if isinstance(candidate, dict):
                request_metadata.update(candidate)

        if not media_url:
            return None

        return {
            "fragment_id": fragment_id,
            "user_id": user_id,
            "media_url": media_url,
            "media_type": media_type or "picture",
            "trace_id": trace_id,
            "metadata": request_metadata,
        }

    def _build_messages(self, media_url: str, media_type: str) -> List[Dict[str, Any]]:
        # 根据媒体类型选择提示词
        is_video = _is_probably_video(media_type, media_url)
        description = PROMPT_VIDEO if is_video else PROMPT_IMAGE
    
        contents: List[Dict[str, Any]] = [{"type": "text", "text": description}]
    
        if is_video:
            # 视频：尝试抽帧并发送多张关键帧；否则回退为单帧/封面
            frames = _sample_video_frames_to_files(media_url, VIDEO_FRAME_COUNT)
            if frames:
                contents.insert(1, {"type": "text", "text": "下面是该视频按顺序抽取的关键帧，请综合所有帧进行客观描述。"})
                total_bytes = 0
                used = 0
                for fp in frames:
                    data_url, raw_len = _file_to_data_url(fp)
                    if total_bytes + raw_len > MAX_TOTAL_IMAGE_BYTES:
                        break
                    contents.append({"type": "image_url", "image_url": {"url": data_url}})
                    total_bytes += raw_len
                    used += 1
                if used == 0:
                    data_url, _ = _file_to_data_url(frames[0])
                    contents.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                contents.insert(1, {"type": "text", "text": "抽帧未启用或失败，以下为视频封面/单帧，请给出概括性描述。"})
                contents.append({"type": "image_url", "image_url": {"url": media_url}})
        else:
            # 图片：单张输入
            contents.append({"type": "image_url", "image_url": {"url": media_url}})
    
        return [{"role": "user", "content": contents}]


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

    def _build_result_message(
        self,
        *,
        trace_id: str,
        metadata: Optional[Dict[str, Any]],
        fragment_id: str,
        user_id: str,
        media_url: str,
        media_type: str,
        features: Optional[Dict[str, Any]],
        error: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result_metadata: Dict[str, Any] = dict(metadata or {})
        if fragment_id:
            result_metadata.setdefault("fragment_id", fragment_id)
        if user_id:
            result_metadata.setdefault("user_id", user_id)
        if media_url:
            result_metadata.setdefault("media_url", media_url)
        if media_type:
            result_metadata.setdefault("media_type", media_type)

        status = RESULT_STATUS_SUCCESS if features else RESULT_STATUS_FAILURE
        error_payload = None
        if status == RESULT_STATUS_FAILURE:
            error_payload = {
                "code": str((error or {}).get("code") or DEFAULT_ERROR_CODE),
                "message": str((error or {}).get("message") or "多模态预处理失败"),
            }

        return {
            "status": status,
            "trace_id": str(trace_id or ""),
            "metadata": result_metadata,
            "payload": features if features else None,
            "error": error_payload,
        }

    def _publish(self, channel: pika.channel.Channel, message: Dict[str, Any]) -> bool:
        fragment_id = (message.get("metadata") or {}).get("fragment_id")
        trace_id = message.get("trace_id")
        if not FRAGMENT_RESULT_EXCHANGE or not FRAGMENT_RESULT_ROUTING_KEY:
            log.error(
                "未配置 fragment_result_exchange 或 routing_key，无法回传预处理结果 fragmentId=%s traceId=%s",
                fragment_id,
                trace_id,
            )
            return False
        try:
            channel.basic_publish(
                exchange=FRAGMENT_RESULT_EXCHANGE,
                routing_key=FRAGMENT_RESULT_ROUTING_KEY,
                body=json.dumps(message, ensure_ascii=False, default=str).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            log.info(
                "已推送预处理结果 status=%s traceId=%s fragmentId=%s exchange=%s rk=%s",
                message.get("status"),
                trace_id,
                fragment_id,
                FRAGMENT_RESULT_EXCHANGE,
                FRAGMENT_RESULT_ROUTING_KEY,
            )
            return True
        except Exception as exc:
            log.exception("推送预处理结果失败 fragmentId=%s err=%s", fragment_id, exc)
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

            features: Optional[Dict[str, Any]] = None
            error_info: Optional[Dict[str, Any]] = None
            try:
                features = self._call_glm(fragment_id, media_url, media_type)
                if not features:
                    error_info = {
                        "code": "NO_FEATURES",
                        "message": "GLM 未返回有效的多模态特征",
                    }
            except Exception as exc:  # noqa: BLE001
                log.exception("处理碎片失败 fragmentId=%s err=%s", fragment_id, exc)
                error_info = {
                    "code": "UNEXPECTED_EXCEPTION",
                    "message": str(exc),
                }
                features = None

            result_message = self._build_result_message(
                trace_id=info.get("trace_id", ""),
                metadata=info.get("metadata"),
                fragment_id=fragment_id,
                user_id=user_id,
                media_url=media_url,
                media_type=media_type,
                features=features,
                error=error_info,
            )

            if result_message.get("status") != RESULT_STATUS_SUCCESS:
                success = False

            published = self._publish(channel, result_message)
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
            "多模态预处理服务已启动 queue=%s exchange=%s rk=%s",
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
