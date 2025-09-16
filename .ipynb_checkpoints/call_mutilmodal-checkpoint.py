# -*- coding: utf-8 -*-
# 文件：call_mutilmodal.py

import os
import json as pyjson
import logging
import time
from io import BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer

try:
    from zhipuai import ZhipuAI  # 官方 SDK（可选）
except Exception:
    ZhipuAI = None  # 没装也不影响，自动走 REST

from .config import config
from elastic_client import (
    get_client,
    ensure_user_profile_index,
    get_user_profile_index,
)

# ================== 可调参数 ==================
MAX_CAPTION_IMAGES = int(config.get("max_caption_images", 999))   # 生成描述的最大图片数
CAPTION_WORKERS    = int(config.get("caption_workers", 4))       # 并发线程数
HTTP_TOTAL_RETRIES = int(config.get("http_total_retries", 3))
HTTP_BACKOFF       = float(config.get("http_backoff_factor", 0.5))
HTTP_CONNECT_TO    = float(config.get("http_connect_timeout", 3.0))
HTTP_READ_TO       = float(config.get("http_read_timeout", 10.0))
GLM_TIMEOUT        = float(config.get("glm_timeout_sec", 60.0))
GLM_MODEL_NAME     = (str(config.get("glm_model_name", "glm-4-AirX")).strip() or "glm-4-AirX")
GLM_USE_SDK        = bool(config.get("glm_use_sdk", True))
# =================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("multimodal")

MODEL_DIR = config.get("model_dir", "/root/autodl-tmp/models")

def _as_text(v):
    """确保写入 ES 的 text 字段一定是字符串。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return pyjson.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

# ---- HTTP session with retry（用于下载图片/REST 调 GLM） ----
def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=HTTP_TOTAL_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

HTTP = make_session()

# ---- 模型只加载一次 ----
blip_path = os.path.join(MODEL_DIR, "blip2-opt-2.7b")
processor = Blip2Processor.from_pretrained(blip_path)
blip_model = Blip2ForConditionalGeneration.from_pretrained(blip_path).to("cuda").eval()

EMBED_MODEL = SentenceTransformer(
    os.path.join(MODEL_DIR, "bge-base-zh-v1.5"),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# ---- ZhipuAI 客户端（可选） ----
def _build_zhipu_client() -> Optional["ZhipuAI"]:
    api_key = (
        config.get("zhipuai_api_key")
        or config.get("glm_api_key")
        or os.getenv("ZHIPUAI_API_KEY", "")
    )
    if not api_key:
        log.warning("未配置 zhipuai_api_key / glm_api_key / ZHIPUAI_API_KEY，无法使用 SDK，将走 REST")
        return None
    if ZhipuAI is None:
        log.warning("未安装 zhipuai SDK，将走 REST")
        return None
    # SDK 不保证支持传入 timeout 等参数；因此仍保留 REST 兜底
    return ZhipuAI(api_key=api_key)

def extract_text(user_data: dict) -> str:
    """从 user 字段里提取文本属性，拼接为上下文文本。"""
    user = user_data.get("user") or {}
    field_names = [
        "genderType", "mbtiType", "zodiacType", "birthday",
        "bio", "job", "edu", "region", "pet", "thirdPartyAuth", "lastSyncTime"
    ]
    parts = []
    for f in field_names:
        v = user.get(f)
        if v and v != "Unknown":
            parts.append(f"{f}: {v}")
    return "\n".join(parts)

def extract_image_urls(user_data: dict, max_imgs: int = MAX_CAPTION_IMAGES) -> List[str]:
    """从 fragments 中抽取 Picture 的 URL，按参数截断。"""
    fragments = user_data.get("fragments") or []
    if not isinstance(fragments, list):
        fragments = []
    image_urls = [
        f.get("sourceUrl")
        for f in fragments
        if isinstance(f, dict) and f.get("type") == "Picture" and f.get("sourceUrl")
    ]
    if max_imgs is None or max_imgs <= 0:
        return image_urls
    return image_urls[:max_imgs]

@lru_cache(maxsize=2048)
def caption_from_url(url: str) -> str:
    """下载图片并用 BLIP2 生成描述。出错返回'无图像内容'。"""
    try:
        resp = HTTP.get(url, timeout=(HTTP_CONNECT_TO, HTTP_READ_TO))
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(blip_model.device)
            out = blip_model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                repetition_penalty=1.5,
                length_penalty=1.0,
            )
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return caption or "无图像内容"
    except Exception as e:
        log.warning("图片处理失败 url=%s err=%s", url, e)
        return "无图像内容"

def batch_make_captions(urls: List[str]) -> List[str]:
    if not urls:
        return []
    captions: List[str] = []
    with ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as ex:
        futures = {ex.submit(caption_from_url, u): u for u in urls}
        for fut in as_completed(futures):
            try:
                captions.append(fut.result())
            except Exception as e:
                log.warning("caption 线程异常: %s", e)
                captions.append("无图像内容")
    return captions

def _extract_json_obj(text: str) -> Dict:
    """从文本中尽量截出最外层 JSON 对象并解析为 dict。失败返回 {}。"""
    if not isinstance(text, str):
        return {}
    start = text.find("{")
    end = text.rfind("}") + 1
    json_str = text[start:end] if start != -1 and end > start else "{}"
    try:
        data = pyjson.loads(json_str)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _glm_call_via_rest(prompt: str) -> Optional[str]:
    """
    直接调用 REST 接口，显式 timeout，避免 SDK 卡住。
    """
    api_key = (
        config.get("zhipuai_api_key")
        or config.get("glm_api_key")
        or os.getenv("ZHIPUAI_API_KEY", "")
    )
    if not api_key:
        log.error("REST 调用失败：未配置 ZHIPUAI API KEY")
        return None

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    payload = {
        "model": GLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        # "response_format": {"type": "json_object"},  # 若服务端不支持请注释
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    try:
        resp = HTTP.post(url, headers=headers, json=payload, timeout=(HTTP_CONNECT_TO, GLM_TIMEOUT))
    except Exception as e:
        log.error("REST 调 GLM 异常: %s", e)
        return None
    cost = (time.time() - t0) * 1000

    if resp.status_code != 200:
        log.error("REST 调 GLM 失败 status=%s cost=%.1fms text[:200]=%s",
                  resp.status_code, cost, resp.text[:200])
        return None

    try:
        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        log.info("REST 调 GLM 成功 cost=%.1fms", cost)
        return content
    except Exception:
        log.error("REST 返回解析失败 text[:200]=%s", resp.text[:200])
        return None

def call_glm4(user_id: str, texts: str, captions: List[str]) -> Dict:
    """
    调用 GLM：优先 SDK；失败或不可用时降级 REST（带显式超时）。
    返回卡片字段字典。
    """
    full_prompt = f"""
你是一个心理学专家，正在分析一位用户的公开内容（图像描述和文字）来构建一个结构化用户卡片。
请根据以下内容，输出一个严格的 JSON 对象，包含以下字段：
traits, writing_style, emotional_patterns, default_needs, attachment_style, profile_text, one_sentence_summary。

【文本内容】：
{texts}

【图像内容】：
{"；".join(captions)}

请用中文回答：
- 必须返回标准 JSON 对象，不能包含任何多余内容（如说明文字、前后缀、markdown代码块、```json 等）；
- 不允许添加注释（如 // 或 #）；
- 字段值如果信息不完整，请根据上下文合理推测填写，避免留空；
- profile_text 是一段简洁自然语言简介；
- one_sentence_summary 是一句极简总结，格式为“一个xxx的xxx”，不需要标点；
""".strip()

    content: Optional[str] = None

    # 1) SDK 路径（可选）
    if GLM_USE_SDK:
        client = _build_zhipu_client()
        if client is not None:
            try:
                t0 = time.time()
                # 某些 SDK 版本可能不支持 timeout 参数；若不支持会抛异常并走 REST 兜底
                resp = client.chat.completions.create(
                    model=GLM_MODEL_NAME,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.0,
                    # response_format={"type": "json_object"},  # 若不支持请注释
                )
                cost = (time.time() - t0) * 1000
                try:
                    content = (resp.choices or [])[0].message.content
                except Exception:
                    content = getattr(resp, "content", None)
                log.info("SDK 调 GLM 成功 userId=%s cost=%.1fms", user_id, cost)
            except Exception as e:
                log.error("SDK 调 GLM 异常，降级 REST：%s", e)

    # 2) REST 兜底
    if not content:
        content = _glm_call_via_rest(full_prompt)

    if not content or not isinstance(content, str):
        log.error("GLM 返回无内容或非字符串 userId=%s", user_id)
        return {}

    data = _extract_json_obj(content)
    if not data:
        log.error("GLM 内容解析为空 userId=%s content[:200]=%s", user_id, content[:200])

    return {
        "traits": data.get("traits", ""),
        "writing_style": data.get("writing_style", ""),
        "emotional_patterns": data.get("emotional_patterns", ""),
        "default_needs": data.get("default_needs", ""),
        "attachment_style": data.get("attachment_style", ""),
        "profile_text": data.get("profile_text", ""),
        "one_sentence_summary": data.get("one_sentence_summary", ""),
    }

def multimodal_analyze_and_save(user_data: dict) -> bool:
    """
    主流程：文本/图片 -> 调 GLM -> 写 ES。
    """
    if not isinstance(user_data, dict):
        log.error("multimodal: 入参 user_data 非 dict，跳过")
        return False

    user = user_data.get("user") or {}
    if not isinstance(user, dict) or not user:
        log.error("multimodal: 缺少 user 字段或为空，跳过。keys=%s", list(user_data.keys()))
        return False

    user_id = str(user.get("userId") or user.get("id") or "").strip()
    if not user_id:
        log.error("multimodal: 缺少 userId，跳过")
        return False

    nick_name = user.get("nickName", "") or ""
    log.info("开始处理用户 userId=%s nick=%s", user_id, nick_name)

    # —— 文本抽取
    t0 = time.time()
    try:
        texts = extract_text({"user": user})
    except Exception as e:
        log.exception("multimodal: 提取文本失败: %s", e)
        texts = ""
    t_text = (time.time() - t0) * 1000

    # —— 图片 caption
    fragments = user_data.get("fragments") or []
    if not isinstance(fragments, list):
        fragments = []
    safe_payload = {"user": user, "fragments": fragments}

    image_urls = extract_image_urls(safe_payload, max_imgs=MAX_CAPTION_IMAGES)
    t1 = time.time()
    captions = batch_make_captions(image_urls)
    while len(captions) < 3:
        captions.append("无图像内容")
    t_caption = (time.time() - t1) * 1000

    log.info("图片描述生成完毕，共 %d 条，开始调用 GLM4 | text=%.1fms caption=%.1fms",
             len(captions), t_text, t_caption)

    # —— 调 GLM
    t2 = time.time()
    card = call_glm4(user_id, texts, captions) or {}
    t_glm = (time.time() - t2) * 1000

    result_row = {
        "name": str(user_id),
        "nickName": nick_name or "",
        "traits": _as_text(card.get("traits", "")),
        "writing_style": _as_text(card.get("writing_style", "")),
        "emotional_patterns": _as_text(card.get("emotional_patterns", "")),
        "default_needs": _as_text(card.get("default_needs", "")),
        "attachment_style": _as_text(card.get("attachment_style", "")),
        "profile_text": _as_text(card.get("profile_text", "")),
        "one_sentence_summary": _as_text(card.get("one_sentence_summary", "")),
    }

    # —— ES 客户端
    try:
        ensure_user_profile_index()
        es = get_client()
    except Exception as e:
        log.exception("ES 客户端初始化失败: %s", e)
        return False

    embed_text = (
        f"角色名：{result_row['name']} "
        f"简介：{result_row['profile_text']} "
        f"核心特质：{result_row['traits']} "
        f"典型需求：{result_row['default_needs']} "
        f"情感模式：{result_row['emotional_patterns']} "
        f"写作风格：{result_row['writing_style']} "
        f"依恋类型：{result_row['attachment_style']} "
        f"推荐语：{result_row['one_sentence_summary']}"
    )

    # —— 向量
    t3 = time.time()
    try:
        embedding = EMBED_MODEL.encode(embed_text, normalize_embeddings=True).tolist()
    except Exception as e:
        log.exception("Embedding 失败，降级为空向量: %s", e)
        embedding = []
    t_embed = (time.time() - t3) * 1000

    es_doc = {
        "userId": user_id,
        "nickName": nick_name,
        "genderType": user.get("genderType"),
        "mbtiType": user.get("mbtiType"),
        "zodiacType": user.get("zodiacType"),
        "districtName": user.get("districtName"),
        "city": user.get("city"),
        "role": user.get("role"),
        "extraRoles": user.get("extraRoles", []),
        "profession": user.get("profession", []),
        "region": user.get("region", []),
        "school": user.get("school", []),
        "embedding": embedding,
        **result_row,
    }

    # —— 写 ES
    t4 = time.time()
    try:
        es.index(
            index=get_user_profile_index(),
            id=user_id,
            document=es_doc,
        )
        t_es = (time.time() - t4) * 1000
        total = t_text + t_caption + t_glm + t_embed + t_es
        log.info(
            "用户 %s 处理完成并写入 ES | text=%.1fms caption=%.1fms glm=%.1fms embed=%.1fms es=%.1fms total=%.1fms",
            user_id, t_text, t_caption, t_glm, t_embed, t_es, total
        )
        return True
    except Exception as e:
        log.exception("写入 ES 失败 userId=%s err=%s", user_id, e)
        return False
