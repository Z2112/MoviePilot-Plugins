import json
import re
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

import openai
import google.generativeai as genai
from groq import Groq

# 使用 MoviePilot 内置的缓存系统
from app.core.cache import TTLCache
from app.core.config import settings
from app.log import logger

# --- 会话缓存 ---
LLMSessionCache = TTLCache(region="llm_service", maxsize=200, ttl=3600)

# --- 工厂函数 ---
def get_llm_client(config: Dict[str, Any]):
    service = config.get("llm_service", "chatgpt")
    try:
        if service == 'chatgpt':
            return OpenAIClient(config)
        elif service == 'gemini':
            return GeminiClient(config)
        elif service == 'grok':
            return GrokClient(config)
    except Exception as e:
        logger.error(f"初始化 {service.upper()} 客户端时出错: {e}")
    return None

# --- 客户端基类 ---
class LLMBaseClient(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt = config.get("customize_prompt")

    @abstractmethod
    def get_response(self, text: str, userid: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_media_name(self, filename: str) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def is_api_error(response: Any) -> Tuple[bool, str]:
        if isinstance(response, dict) and "errorMsg" in response:
            return True, response["errorMsg"]
        if isinstance(response, str) and "API 调用失败" in response:
            return True, response
        return False, ""

    def _parse_json_from_response(self, content: str) -> Dict[str, Any]:
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content.strip(), re.DOTALL)
            json_str = match.group(1) if match else content
            return json.loads(json_str)
        except Exception:
            return {"errorMsg": "返回内容不是有效的JSON格式", "content": content}

# --- OpenAI/ChatGPT 客户端 ---
class OpenAIClient(LLMBaseClient):
    def __init__(self,
