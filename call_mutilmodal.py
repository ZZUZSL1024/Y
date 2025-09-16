# -*- coding: utf-8 -*-
"""用户画像生成服务：读取预处理好的多模态特征并构建用户画像。"""

import os
import json as pyjson
import logging
import time
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer

from .config import config
from elastic_client import (
    get_client,
    ensure_user_profile_index,
    get_user_profile_index,
)

from .glm_client import GLMClient, extract_json_object

# ================== 配置 ==================
MAX_CAPTION_IMAGES = int(config.get("max_caption_images", 999))   # 读取描述的最大图片数
GLM_TIMEOUT = float(config.get("glm_timeout_sec", 60.0))
GLM_MODEL_NAME = (str(config.get("glm_model_name", "glm-4.5v")).strip() or "glm-4.5v")
MODEL_DIR = config.get("model_dir", "/root/autodl-tmp/models")
ES_CONFIG = config.get("elasticsearch", {}) or {}
FRAGMENT_INDEX = str(
    config.get("fragment_index")
    or ES_CONFIG.get("fragment_index")
    or ES_CONFIG.get("fragments_index")
    or "user_fragments"
).strip()
FRAGMENT_USER_FIELD = str(config.get("fragment_user_field", "userId")).strip() or "userId"
FRAGMENT_MULTIMODAL_FIELD = (
    str(config.get("fragment_multimodal_field", "multimodal_features")).strip()
    or "multimodal_features"
)
GLM_API_KEY = (
    config.get("zhipuai_api_key")
    or config.get("glm_api_key")
    or os.getenv("ZHIPUAI_API_KEY", "")
)
GLM_BASE_URL = (
    str(config.get("glm_api_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions")).strip()
    or "https://open.bigmodel.cn/api/paas/v4/chat/completions"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("multimodal")

# ---- 嵌入模型只加载一次 ----
EMBED_MODEL = SentenceTransformer(
    os.path.join(MODEL_DIR, "bge-base-zh-v1.5"),
    device="cuda" if torch.cuda.is_available() else "cpu",
)

_GLM_CLIENT: Optional[GLMClient] = None


def get_glm_client() -> Optional[GLMClient]:
    """懒加载 GLM 客户端，避免重复创建 Session。"""
    global _GLM_CLIENT
    if _GLM_CLIENT is None:
        _GLM_CLIENT = GLMClient(
            api_key=GLM_API_KEY,
            model_name=GLM_MODEL_NAME,
            base_url=GLM_BASE_URL,
            timeout=GLM_TIMEOUT,
        )
    return _GLM_CLIENT


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _as_text(value) -> str:
    """确保写入 ES 的字段为字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return pyjson.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def extract_text(user_data: dict) -> str:
    """从 user 字段里提取文本属性，拼接为上下文文本。"""
    user = user_data.get("user") or {}
    field_names = [
        "genderType",
        "mbtiType",
        "zodiacType",
        "birthday",
        "bio",
        "job",
        "edu",
        "region",
        "pet",
        "thirdPartyAuth",
        "lastSyncTime",
    ]
    parts = []
    for field in field_names:
        value = user.get(field)
        if value and value != "Unknown":
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def _normalize_captions(captions: List[str], minimum: int = 3) -> List[str]:
    """去重、截断并保证最少条数。"""
    limit = MAX_CAPTION_IMAGES if MAX_CAPTION_IMAGES > 0 else None
    cleaned: List[str] = []
    seen = set()
    for caption in captions:
        if not isinstance(caption, str):
            continue
        text = caption.strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if limit is not None and len(cleaned) >= limit:
            break
    while len(cleaned) < minimum:
        cleaned.append("无图像内容")
    return cleaned


def _compose_caption_from_features(features: Optional[Dict]) -> Optional[str]:
    if not isinstance(features, dict):
        return None
    description = features.get("description")
    if isinstance(description, str):
        description = description.strip()
    else:
        description = None
    tags = features.get("tags")
    tag_text = ""
    if isinstance(tags, (list, tuple)):
        tag_items = [str(tag).strip() for tag in tags if str(tag).strip()]
        if tag_items:
            tag_text = " 标签：" + "、".join(tag_items)
    if description:
        return f"{description}{tag_text}" if tag_text else description
    if tag_text:
        return tag_text.replace(" 标签：", "标签：")
    return None


def _build_fragment_query(user_id: str) -> Dict:
    fields = []
    if FRAGMENT_USER_FIELD:
        fields.append(FRAGMENT_USER_FIELD)
        if not FRAGMENT_USER_FIELD.endswith(".keyword"):
            fields.append(f"{FRAGMENT_USER_FIELD}.keyword")
    fields.extend(["userId", "userId.keyword"])
    # 去重同时保持顺序
    seen_fields = []
    for field in fields:
        if field and field not in seen_fields:
            seen_fields.append(field)

    should_terms = [{"term": {field: {"value": user_id}}} for field in seen_fields]
    exists_conditions = [
        {"exists": {"field": f"{FRAGMENT_MULTIMODAL_FIELD}.description"}}
    ]
    alt_field = f"fragment.{FRAGMENT_MULTIMODAL_FIELD}.description"
    if alt_field not in (condition.get("exists", {}).get("field") for condition in exists_conditions):
        exists_conditions.append({"exists": {"field": alt_field}})

    return {
        "bool": {
            "must": [
                {
                    "bool": {
                        "should": should_terms,
                        "minimum_should_match": 1,
                    }
                }
            ],
            "filter": [
                {
                    "bool": {
                        "should": exists_conditions,
                        "minimum_should_match": 1,
                    }
                }
            ],
        }
    }


def fetch_user_fragment_descriptions(es, user_id: str, limit: int = MAX_CAPTION_IMAGES) -> List[str]:
    if not user_id:
        return []
    index = FRAGMENT_INDEX
    if not index:
        log.warning("未配置碎片索引，无法读取多模态描述 userId=%s", user_id)
        return []

    query = _build_fragment_query(user_id)
    source_fields = [
        FRAGMENT_MULTIMODAL_FIELD,
        f"{FRAGMENT_MULTIMODAL_FIELD}.description",
        f"{FRAGMENT_MULTIMODAL_FIELD}.tags",
        "fragment",
    ]
    size = limit if limit and limit > 0 else 100

    try:
        resp = es.search(index=index, query=query, source=source_fields, size=size)
    except TypeError:
        body = {"query": query, "_source": source_fields, "size": size}
        resp = es.search(index=index, body=body)
    except Exception as exc:
        log.exception("查询碎片多模态描述失败 userId=%s err=%s", user_id, exc)
        return []

    hits = (resp.get("hits") or {}).get("hits") or []
    captions: List[str] = []
    for hit in hits:
        source = hit.get("_source") or {}
        caption = _compose_caption_from_features(source.get(FRAGMENT_MULTIMODAL_FIELD))
        if not caption and isinstance(source.get("fragment"), dict):
            caption = _compose_caption_from_features(
                source["fragment"].get(FRAGMENT_MULTIMODAL_FIELD)
            )
        if caption:
            captions.append(caption)
    return captions


def call_glm4(user_id: str, texts: str, captions: List[str]) -> Dict[str, str]:
    """调用 GLM 构建用户画像。"""
    prompt = f"""
你是一个心理学专家，正在分析一位用户的公开内容（图像描述和文字）来构建一个结构化用户卡片。
请根据以下内容，输出一个严格的 JSON 对象，包含以下字段：
traits, writing_style, emotional_patterns, default_needs, attachment_style, profile_text, one_sentence_summary。

【文本内容】：
{texts}

【图像内容】：
{"；".join(captions)}

请用中文回答：
- 必须返回标准 JSON 对象，不能包含任何多余内容（如说明文字、前后缀、markdown 代码块、```json 等）；
- 不允许添加注释（如 // 或 #）；
- 字段值如果信息不完整，请根据上下文合理推测填写，避免留空；
- profile_text 是一段简洁自然语言简介；
- one_sentence_summary 是一句极简总结，格式为“一个xxx的xxx”，不需要标点；
""".strip()

    client = get_glm_client()
    if client is None:
        log.error("GLM 客户端初始化失败，无法生成用户画像 userId=%s", user_id)
        return {}

    t0 = time.time()
    content = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    cost = (time.time() - t0) * 1000

    if not content or not isinstance(content, str):
        log.error("GLM 返回无内容或非字符串 userId=%s cost=%.1fms", user_id, cost)
        return {}

    data = extract_json_object(content)
    if not data:
        log.error(
            "GLM 内容解析为空 userId=%s cost=%.1fms content[:200]=%s",
            user_id,
            cost,
            content[:200],
        )

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
    """主流程：读取预处理结果 -> 调 GLM -> 写入 ES。"""
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

    try:
        es = get_client()
    except Exception as exc:
        log.exception("ES 客户端初始化失败: %s", exc)
        return False

    # —— 文本抽取
    t0 = time.time()
    try:
        texts = extract_text({"user": user})
    except Exception as exc:
        log.exception("multimodal: 提取文本失败: %s", exc)
        texts = ""
    t_text = (time.time() - t0) * 1000

    # —— 获取预处理的图片描述
    t1 = time.time()
    raw_captions = fetch_user_fragment_descriptions(es, user_id, MAX_CAPTION_IMAGES)
    captions = _normalize_captions(raw_captions)
    t_caption = (time.time() - t1) * 1000
    log.info(
        "从预处理服务获取到 %d/%d 条图片描述 userId=%s",
        len(raw_captions),
        len(captions),
        user_id,
    )

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
    except Exception as exc:
        log.exception("Embedding 失败，降级为空向量: %s", exc)
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
        ensure_user_profile_index()
        es.index(
            index=get_user_profile_index(),
            id=user_id,
            document=es_doc,
        )
        t_es = (time.time() - t4) * 1000
        total = t_text + t_caption + t_glm + t_embed + t_es
        log.info(
            "用户 %s 处理完成并写入 ES | text=%.1fms caption=%.1fms glm=%.1fms embed=%.1fms es=%.1fms total=%.1fms",
            user_id,
            t_text,
            t_caption,
            t_glm,
            t_embed,
            t_es,
            total,
        )
        return True
    except Exception as exc:
        log.exception("写入 ES 失败 userId=%s err=%s", user_id, exc)
        return False
