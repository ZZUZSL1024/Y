# -*- coding: utf-8 -*-
"""多模态预处理服务：监听碎片事件并生成结构化描述后回传给后端。"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import re
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import pika
from PIL import Image

# ========= 你的配置模块 =========
from .config import config

# ========= Qwen（DashScope OpenAI 兼容）=========
from openai import OpenAI


# ====== 多模态提示词：缺失补全 ======
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
你是一个专业的视频画面分析引擎。现在会收到按时间顺序抽取的多张关键帧（无音频/时间戳），请综合所有帧进行客观描述。仅返回一个 JSON，对象里只包含 "description" 和 "tags" 两个键：

{
  "description": "按结构化方式书写：A) 全局概括；B) 时间序列要点（帧1…帧N可见变化与动作）；C) 场景/镜头变化；D) 核心主体与交互；E) 视觉风格（光线/色调/运动模糊/景深）；F) 最后一帧状态。避免主观动机与编造信息。",
  "tags": ["稳定元素与关键动作的名词/动名词，如：'室内办公','人群走动','镜头切换','特写','低照度'"]
}
""".strip()


# -----------------------------------------------------------------------------
# OpenAI 兼容客户端（直连 Qwen，不依赖 common/inference.py）
# -----------------------------------------------------------------------------
class QwenClient:
    def __init__(self, api_key: Optional[str], base_url: str, default_vl_model: str) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.default_vl_model = default_vl_model or "qwen-vl-max"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_multimodal(self, messages: List[Dict[str, Any]], model: Optional[str] = None,
                        response_format: Optional[Dict[str, Any]] = None) -> str:
        req = {"model": model or self.default_vl_model, "messages": messages}
        if response_format:
            req["extra_body"] = {"response_format": response_format}
        resp = self.client.chat.completions.create(**req)
        return (resp.choices[0].message.content or "").strip()


# -----------------------------------------------------------------------------
# 常量 & 配置
# -----------------------------------------------------------------------------
QWEN_VL_MODEL = str(config.get("qwen_vl_model") or "qwen-vl-max")
QWEN_BASE_URL = str(config.get("qwen_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_API_KEY = 'sk-fd793db19fdd435697cb7deb35c188f4'

# 审核/错误兜底
MODERATION_FALLBACK_AS_SUCCESS = bool(config.get("moderation_fallback_as_success", True))
MODERATION_PLACEHOLDER_DESC = str(config.get("moderation_placeholder_desc", "内容因平台安全策略或格式问题无法进行视觉分析。"))
MODERATION_PLACEHOLDER_TAGS = list(config.get("moderation_placeholder_tags", ["受限内容"]))

# 隔离留存
MODERATION_SAVE_BLOCKED_MEDIA = bool(config.get("moderation_save_blocked_media", True))
MODERATION_QUARANTINE_DIR = str(config.get("moderation_quarantine_dir", "./quarantine"))
MODERATION_QUARANTINE_MAX_BYTES = int(config.get("moderation_quarantine_max_bytes", 10 * 1024 * 1024))  # 10MB

# 图像预处理参数（新）
IMAGE_MAX_PIXELS = int(config.get("image_max_pixels", 1280 * 720))          # 单张图片最大像素数（约 720p）
IMAGE_MAX_BYTES = int(config.get("image_max_bytes", 2 * 1024 * 1024))       # 单张图片输出 dataURL 前原始字节上限（2MB）
IMAGE_JPEG_QUALITY_START = int(config.get("image_jpeg_quality_start", 90))  # 初始 JPEG 质量
IMAGE_JPEG_QUALITY_MIN = int(config.get("image_jpeg_quality_min", 60))      # 最低 JPEG 质量

# 视频抽帧（本地 -> data URL）
VIDEO_FRAME_COUNT = int(config.get("video_frame_count", 5))
VIDEO_MAX_PIXELS = int(config.get("video_max_pixels", 1280 * 720))
VIDEO_EXTS = set((config.get("video_exts") or ".mp4,.mov,.m4v,.webm,.avi,.mkv").lower().split(","))
MAX_TOTAL_IMAGE_BYTES = int(config.get("video_max_total_image_bytes", 6 * 1024 * 1024))  # 多帧总上限

# RabbitMQ
RABBITMQ_HOST = config["rabbitmq_host"]
RABBITMQ_USER = config["rabbitmq_user"]
RABBITMQ_PASS = config["rabbitmq_pass"]
FRAGMENT_QUEUE_NAME = config.get("fragment_queue_name", "fragment.preprocessing.queue")
FRAGMENT_ROUTING_KEY = config.get("fragment_routing_key", "user.change.fragment")
FRAGMENT_EXCHANGE = config.get("fragment_exchange", "user.behavior.events.exchange")
FRAGMENT_RESULT_EXCHANGE = config.get("fragment_result_exchange", "ai.processing.results.exchange")
FRAGMENT_RESULT_ROUTING_KEY = config.get("fragment_result_routing_key", "fragment.preprocessing.completed")

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


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
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

def _safe_comp(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:120]

def _guess_ext_from_headers(ct: Optional[str], url: str) -> str:
    if ct:
        ct = ct.split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
            "video/webm": ".webm",
        }
        if ct in mapping:
            return mapping[ct]
        ext = mimetypes.guess_extension(ct) or ""
        if ext:
            return ext
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else ".bin"

def _download_bytes(url: str, timeout: float = 10.0) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content, resp.headers.get("Content-Type")
    except Exception as e:
        log.warning("下载失败 url=%s err=%s", url, e)
        return None, None

def _image_bytes_to_jpeg(data: bytes, max_pixels: int = IMAGE_MAX_PIXELS,
                         max_bytes: int = IMAGE_MAX_BYTES) -> Optional[bytes]:
    """
    将任意图片字节尝试转为 RGB JPEG，控制像素上限与体积上限。
    """
    try:
        im = Image.open(io.BytesIO(data))
    except Exception:
        return None

    # 处理 EXIF 方向
    try:
        exif = im.getexif()
        orientation = exif.get(274)
        rotate_map = {3: 180, 6: 270, 8: 90}
        if orientation in rotate_map:
            im = im.rotate(rotate_map[orientation], expand=True)
    except Exception:
        pass

    # 转 RGB
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    elif im.mode == "L":
        im = im.convert("RGB")

    # 降分辨率
    w, h = im.size
    if w * h > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        im = im.resize((nw, nh), Image.LANCZOS)

    # 逐步降质量以满足大小限制
    quality = IMAGE_JPEG_QUALITY_START
    while quality >= IMAGE_JPEG_QUALITY_MIN:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return b
        quality -= 5

    # 最低质量仍超限，再次降分辨率后保存最低质量
    nw, nh = max(1, int(im.size[0] * 0.8)), max(1, int(im.size[1] * 0.8))
    im = im.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY_MIN, optimize=True)
    return buf.getvalue()

def _bytes_to_data_url(b: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"

def _image_url_to_data_url(media_url: str) -> Tuple[Optional[str], Optional[bytes]]:
    raw, ct = _download_bytes(media_url)
    if not raw:
        return None, None
    jpeg = _image_bytes_to_jpeg(raw)
    if not jpeg:
        return None, None
    return _bytes_to_data_url(jpeg, "image/jpeg"), jpeg


def _file_to_data_url(fp: str) -> Tuple[str, int]:
    with open(fp, "rb") as f:
        raw = f.read()
    return _bytes_to_data_url(raw, "image/jpeg"), len(raw)

def _sample_video_frames_to_files(media_url: str, num_frames: int = VIDEO_FRAME_COUNT) -> List[str]:
    if not _has_ffmpeg():
        return []
    tmpdir = tempfile.mkdtemp(prefix="vidframes_")
    out_pattern = os.path.join(tmpdir, "frame_%03d.jpg")
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
    frames = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".jpg")])
    return frames[: max(1, num_frames)] if frames else []

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    depth = 0
    in_string = False
    esc = False
    begin = None
    for i, ch in enumerate(text):
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch in ('"', "'"):
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
            elif ch == "{":
                if depth == 0:
                    begin = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and begin is not None:
                    snippet = text[begin : i + 1]
                    try:
                        obj = json.loads(snippet)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
    return None

def _save_quarantine(
    media_url: str,
    fragment_id: str,
    err_code: str,
    request_id: Optional[str],
    prepared_bytes: Optional[bytes] = None,   # 新增：可选的已重编码JPEG
) -> Optional[str]:
    if not MODERATION_SAVE_BLOCKED_MEDIA:
        return None
    os.makedirs(MODERATION_QUARANTINE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = _safe_comp(request_id or "no_reqid")
    fid = _safe_comp(fragment_id or "no_fid")

    # 1. 保存原图（直链下载）
    media_path = None
    total = 0
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            with client.stream("GET", media_url) as resp:
                resp.raise_for_status()
                ext = _guess_ext_from_headers(resp.headers.get("Content-Type"), media_url)
                media_name = f"{ts}_fid{fid}_{err_code}_{rid}_orig{ext}"
                media_path = os.path.join(MODERATION_QUARANTINE_DIR, media_name)
                with open(media_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MODERATION_QUARANTINE_MAX_BYTES:
                            f.write(chunk[: MODERATION_QUARANTINE_MAX_BYTES - (total - len(chunk))])
                            break
                        f.write(chunk)
    except Exception as exc:
        log.error("保存隔离样本原图失败 url=%s err=%s", media_url, exc)

    # 2. 保存已重编码 JPEG（如果有）
    prepared_path = None
    if prepared_bytes:
        try:
            prepared_name = f"{ts}_fid{fid}_{err_code}_{rid}_prepared.jpg"
            prepared_path = os.path.join(MODERATION_QUARANTINE_DIR, prepared_name)
            with open(prepared_path, "wb") as pf:
                pf.write(prepared_bytes[:MODERATION_QUARANTINE_MAX_BYTES])
        except Exception as exc:
            log.error("保存隔离样本重编码JPEG失败 url=%s err=%s", media_url, exc)

    # 3. 写 meta
    try:
        meta = {
            "fragment_id": fragment_id,
            "request_id": request_id,
            "error_code": err_code,
            "url": media_url,
            "saved_at": _now_iso(),
            "orig_path": media_path,
            "prepared_path": prepared_path,
            "size_bytes_capped": min(total, MODERATION_QUARANTINE_MAX_BYTES) if total else None,
        }
        meta_name = f"{ts}_fid{fid}_{err_code}_{rid}_meta.json"
        meta_path = os.path.join(MODERATION_QUARANTINE_DIR, meta_name)
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        log.info("已保存隔离样本 meta=%s", meta_path)
    except Exception as exc:
        log.error("保存隔离样本 meta 失败 url=%s err=%s", media_url, exc)

    return media_path or prepared_path



MEDIA_URL_FIELDS = ["mediaUrl", "sourceUrl", "url", "coverUrl", "thumbnailUrl"]
MEDIA_TYPE_FIELDS = ["mediaType", "type", "fragmentType", "category"]


# -----------------------------------------------------------------------------
# 主服务
# -----------------------------------------------------------------------------
class MultimodalPreprocessingService:
    def __init__(self) -> None:
        self.qwen = QwenClient(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            default_vl_model=QWEN_VL_MODEL,
        )

    # 解析消息
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

    # 构造消息（返回 messages, used_data_url, raw_image_url）
    def _build_messages(self, media_url: str, media_type: str) -> Tuple[List[Dict[str, Any]], bool, Optional[str], Optional[bytes]]:
        is_video = _is_probably_video(media_type, media_url)
        description = PROMPT_VIDEO if is_video else PROMPT_IMAGE
        contents: List[Dict[str, Any]] = [{"type": "text", "text": description}]
        used_data_url = False
        raw_image_url = None
        prepared_bytes: Optional[bytes] = None
    
        if is_video:
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
                used_data_url = True
            else:
                contents.insert(1, {"type": "text", "text": "抽帧未启用或失败，以下为视频封面/单帧，请给出概括性描述。"})
                data_url, prep = _image_url_to_data_url(media_url)
                if data_url:
                    contents.append({"type": "image_url", "image_url": {"url": data_url}})
                    used_data_url = True
                    prepared_bytes = prep
                else:
                    contents.append({"type": "image_url", "image_url": {"url": media_url}})
                    raw_image_url = media_url
        else:
            data_url, prep = _image_url_to_data_url(media_url)
            if data_url:
                contents.append({"type": "image_url", "image_url": {"url": data_url}})
                used_data_url = True
                prepared_bytes = prep
            else:
                contents.append({"type": "image_url", "image_url": {"url": media_url}})
                raw_image_url = media_url
    
        return [{"role": "user", "content": contents}], used_data_url, raw_image_url, prepared_bytes

    def _parse_openai_error(self, exc: Exception) -> Dict[str, Any]:
        info: Dict[str, Any] = {"raw": str(exc)}
        # 尝试抓取 JSON
        # 1) OpenAI 异常可能有 response 对象
        try:
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    js = resp.json()
                    if isinstance(js, dict):
                        info.update(js)
                        return info
                except Exception:
                    pass
        except Exception:
            pass
        # 2) 从字符串切 JSON
        s = str(exc)
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(s[start : end + 1].replace("'", '"'))
                if isinstance(obj, dict):
                    info.update(obj)
            except Exception:
                pass
        return info

    def _call_qwen(self, fragment_id: str, media_url: str, media_type: str) -> Optional[Dict[str, Any]]:
        messages, used_data_url, raw_image_url, prepared_bytes = self._build_messages(media_url, media_type)
        start = time.time()
        try:
            content = self.qwen.chat_multimodal(
                messages=messages,
                model=QWEN_VL_MODEL,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            cost = (time.time() - start) * 1000
            info = self._parse_openai_error(e)
            err_obj = info.get("error") or {}
            err_code = err_obj.get("code") or ""
            req_id = info.get("id") or info.get("request_id")
    
            log.error(
                "Qwen 多模态接口异常 fragmentId=%s cost=%.1fms code=%s request_id=%s err=%s",
                fragment_id, cost, err_code or None, req_id or None, info or {}
            )
    
            # —— 安全审查拦截：落盘 + 可选兜底 ——
            if err_code == "data_inspection_failed":
                _save_quarantine(media_url, fragment_id, err_code, req_id, prepared_bytes=prepared_bytes)
                if MODERATION_FALLBACK_AS_SUCCESS:
                    return {
                        "description": MODERATION_PLACEHOLDER_DESC,
                        "tags": MODERATION_PLACEHOLDER_TAGS,
                        "model_version": QWEN_VL_MODEL,
                        "processed_at": _now_iso(),
                        "raw_response": {"error_code": err_code, "request_id": req_id},
                        "moderation_blocked": True,
                    }
                return None
    
            # —— 非法图片：若首轮不是 dataURL 且有直链，强制重编码重试；不论结果，先落盘出错样本 ——
            if err_code == "invalid_parameter_error" and (not used_data_url) and raw_image_url:
                _save_quarantine(media_url, fragment_id, err_code, req_id, prepared_bytes=None)
                forced_data_url, forced_bytes = _image_url_to_data_url(raw_image_url)
                if forced_data_url:
                    retry_messages = [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT_IMAGE if not _is_probably_video(media_type, media_url) else PROMPT_VIDEO},
                        {"type": "image_url", "image_url": {"url": forced_data_url}},
                    ]}]
                    try:
                        content = self.qwen.chat_multimodal(
                            messages=retry_messages,
                            model=QWEN_VL_MODEL,
                            response_format={"type": "json_object"},
                        )
                    except Exception as e2:
                        info2 = self._parse_openai_error(e2)
                        err_code2 = (info2.get("error") or {}).get("code") or "invalid_parameter_error"
                        req_id2 = info2.get("id") or info2.get("request_id")
                        _save_quarantine(media_url, fragment_id, err_code2, req_id2, prepared_bytes=forced_bytes)
                        return None
                else:
                    _save_quarantine(media_url, fragment_id, "invalid_image_format_download_failed", req_id, prepared_bytes=None)
                    return None
            else:
                # —— 其他所有错误：一律落盘（含原图与重编码）——
                _save_quarantine(media_url, fragment_id, err_code or "call_failed", req_id, prepared_bytes=prepared_bytes)
                return None
    
        # 正常解析
        if not content:
            _save_quarantine(media_url, fragment_id, "empty_response", None, prepared_bytes=prepared_bytes)
            log.error("Qwen 返回空结果 fragmentId=%s", fragment_id)
            return None
    
        data = extract_json_object(content)
        if not data:
            _save_quarantine(media_url, fragment_id, "parse_json_failed", None, prepared_bytes=prepared_bytes)
            log.error("Qwen 返回结果无法解析为 JSON fragmentId=%s content[:200]=%s", fragment_id, content[:200])
            return None
    
        description = (data.get("description") or data.get("caption") or "").strip()
        tags = _ensure_list(data.get("tags") or data.get("keywords"))
        if (not description) and tags:
            description = "、".join(tags[:5])
    
        return {
            "description": description,
            "tags": tags,
            "model_version": QWEN_VL_MODEL,
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
                fragment_id, trace_id,
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
                message.get("status"), trace_id, fragment_id, FRAGMENT_RESULT_EXCHANGE, FRAGMENT_RESULT_ROUTING_KEY,
            )
            return True
        except Exception as exc:
            log.exception("推送预处理结果失败 fragmentId=%s err=%s", fragment_id, exc)
            return False

    # 消费处理
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
                log.warning("消息缺少必要字段，跳过：%s", item)
                continue
            fragment_id = info["fragment_id"]
            media_url = info["media_url"]
            media_type = info["media_type"]
            user_id = info["user_id"]

            features: Optional[Dict[str, Any]] = None
            error_info: Optional[Dict[str, Any]] = None
            try:
                features = self._call_qwen(fragment_id, media_url, media_type)
                if not features:
                    error_info = {"code": "NO_FEATURES", "message": "Qwen 未返回有效多模态特征"}
            except Exception as exc:  # noqa: BLE001
                log.exception("处理碎片失败 fragmentId=%s err=%s", fragment_id, exc)
                error_info = {"code": "UNEXPECTED_EXCEPTION", "message": str(exc)}
                features = None

            message = self._build_result_message(
                trace_id=info.get("trace_id", ""),
                metadata=info.get("metadata"),
                fragment_id=fragment_id,
                user_id=user_id,
                media_url=media_url,
                media_type=media_type,
                features=features,
                error=error_info,
            )

            if message.get("status") != RESULT_STATUS_SUCCESS:
                success = False
            if not self._publish(channel, message):
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
            client_properties={"connection_name": f"{PROJECT_NAME}-fragment-worker@{CLIENT_INFO}-{os.getpid()}"},
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        try:
            channel.queue_declare(queue=FRAGMENT_QUEUE_NAME, durable=True, passive=True)
        except pika.exceptions.ChannelClosedByBroker:
            channel = connection.channel()
            raise RuntimeError(f"Queue {FRAGMENT_QUEUE_NAME} not found. 请先在 RabbitMQ 创建该队列。")

        channel.queue_bind(exchange=FRAGMENT_EXCHANGE, queue=FRAGMENT_QUEUE_NAME, routing_key=FRAGMENT_ROUTING_KEY)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=FRAGMENT_QUEUE_NAME, on_message_callback=self._handle, auto_ack=False)

        log.info(
            "多模态预处理服务已启动 queue=%s exchange=%s rk=%s model=%s base_url=%s quarantine_dir=%s save_blocked=%s",
            FRAGMENT_QUEUE_NAME, FRAGMENT_EXCHANGE, FRAGMENT_ROUTING_KEY,
            QWEN_VL_MODEL, QWEN_BASE_URL, MODERATION_QUARANTINE_DIR, MODERATION_SAVE_BLOCKED_MEDIA,
        )
        channel.start_consuming()


def start_multimodal_preprocessor() -> None:
    MultimodalPreprocessingService().start()


if __name__ == "__main__":
    start_multimodal_preprocessor()
