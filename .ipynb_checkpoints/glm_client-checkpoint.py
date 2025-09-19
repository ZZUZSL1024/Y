# -*- coding: utf-8 -*-
"""统一的 GLM-4.5v 客户端封装。

该模块负责处理鉴权、请求构造、异常重试与基础的结果解析，
供多模态预处理服务与用户画像服务复用。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Sequence, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import config

log = logging.getLogger("glm-client")

# ---- 默认重试/超时配置（可通过 config 覆盖） ----
HTTP_TOTAL_RETRIES = int(config.get("http_total_retries", 3))
HTTP_BACKOFF = float(config.get("http_backoff_factor", 0.5))
HTTP_CONNECT_TO = float(config.get("http_connect_timeout", 3.0))
HTTP_READ_TO = float(config.get("http_read_timeout", 10.0))


def _build_session(total_retries: int = HTTP_TOTAL_RETRIES,
                   backoff_factor: float = HTTP_BACKOFF) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _default_base_url() -> str:
    url = str(config.get("glm_api_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions"))
    return url.strip() or "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class GLMClient:
    """轻量封装智谱 GLM Chat Completion 接口。"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "glm-4.5v",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        connect_timeout: Optional[float] = None,
        read_timeout: Optional[float] = None,
        total_retries: int = HTTP_TOTAL_RETRIES,
        backoff_factor: float = HTTP_BACKOFF,
    ) -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model_name.strip() or "glm-4.5v"
        self.base_url = (base_url or _default_base_url()).strip()
        self.timeout = timeout
        self.connect_timeout = connect_timeout or HTTP_CONNECT_TO
        self.read_timeout = read_timeout or timeout or HTTP_READ_TO
        self.session = _build_session(total_retries=total_retries, backoff_factor=backoff_factor)

    # ---- 请求核心 ----
    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        调用 Chat Completion 接口，返回第一条 choice 的 content。

        :param messages: Chat Completions 格式的消息数组。
        :param temperature: 采样温度，默认 0.0，保证结果稳定。
        :param response_format: 可选的响应结构约束，例如 {"type": "json_object"}。
        :param extra_payload: 扩展字段透传。
        """

        if not self.api_key:
            log.error("GLMClient 未配置 API Key，无法发起请求")
            return None

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        t0 = time.time()
        try:
            resp = self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=(self.connect_timeout, self.read_timeout),
            )
        except Exception as exc:  # 网络异常
            cost = (time.time() - t0) * 1000
            log.exception("GLMClient 请求异常 cost=%.1fms err=%s", cost, exc)
            return None

        cost = (time.time() - t0) * 1000
        if resp.status_code != 200:
            log.error(
                "GLMClient 返回非 200 status=%s cost=%.1fms text[:200]=%s",
                resp.status_code,
                cost,
                resp.text[:200],
            )
            return None

        try:
            data = resp.json()
        except json.JSONDecodeError:
            log.error("GLMClient 返回无法解析为 JSON cost=%.1fms text[:200]=%s", cost, resp.text[:200])
            return None

        try:
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        except Exception:  # 结构异常
            content = None

        if content:
            log.info("GLMClient 调用成功 model=%s cost=%.1fms", self.model_name, cost)
        else:
            log.warning("GLMClient 调用完成但未获取到内容 model=%s cost=%.1fms", self.model_name, cost)

        return content


def extract_json_object(content: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
    """尝试从模型返回的字符串中解析 JSON 对象。"""

    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}

    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end <= start:
        return {}

    json_str = content[start:end]
    try:
        data = json.loads(json_str)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

