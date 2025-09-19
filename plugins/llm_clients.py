import json
import re
import time
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

import openai
import google.generativeai as genai
from groq import Groq
from cacheout import Cache
from app.core.config import settings
from app.log import logger

# --- 会话缓存 ---
LLMSessionCache = Cache(maxsize=200, ttl=3600, timer=time.time, default=None)

# --- 工厂函数 ---
def get_llm_client(config: Dict[str, Any]):
    """
    根据配置获取相应的 LLM 客户端实例
    """
    service = config.get("llm_service", "chatgpt")
    try:
        if service == 'chatgpt':
            return OpenAIClient(config)
        elif service == 'gemini':
            return GeminiClient(config)
        elif service == 'grok':
            return GrokClient(config)
        else:
            logger.error(f"未知的 LLM 服务: {service}")
            return None
    except Exception as e:
        logger.error(f"初始化 {service.upper()} 客户端时出错: {e}")
        return None

# --- 客户端基类 ---
class LLMBaseClient(ABC):
    """
    所有 LLM 客户端的抽象基类
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = ""
        self.prompt = config.get("customize_prompt")

    @abstractmethod
    def get_response(self, text: str, userid: str) -> str:
        """获取聊天对话响应"""
        raise NotImplementedError

    @abstractmethod
    def get_media_name(self, filename: str) -> Dict[str, Any]:
        """从文件名中提取媒体信息"""
        raise NotImplementedError

    @staticmethod
    def is_api_error(response: Any) -> Tuple[bool, str]:
        """判断响应是否为API错误"""
        if isinstance(response, dict) and response.get("errorMsg"):
            return True, response.get("errorMsg")
        if isinstance(response, str) and "API 调用失败" in response:
            return True, response
        return False, ""

    def _parse_media_name_from_response(self, content: str) -> Dict[str, Any]:
        """从模型返回的内容中解析出媒体信息JSON"""
        try:
            # 移除Markdown代码块标记
            pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(pattern, content.strip(), re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = content
            return json.loads(json_str)
        except json.JSONDecodeError:
             logger.error(f"无法将模型返回内容解析为JSON: {content}")
             return {
                "content": content,
                "errorMsg": "返回内容不是有效的JSON格式"
            }
        except Exception as e:
            logger.error(f"解析媒体名称时发生未知错误: {e}")
            return {
                "content": content,
                "errorMsg": str(e)
            }

# --- OpenAI/ChatGPT 客户端 ---
class OpenAIClient(LLMBaseClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get("openai_key")
        api_url = config.get("openai_url")
        if not api_key or not api_url:
            raise ValueError("OpenAI API密钥和URL不能为空")

        self.model = config.get("model", "gpt-4o")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=f"{api_url}/v1" if not config.get("compatible") else api_url,
            http_client=settings.NET_PROXY if config.get("proxy") else None
        )
    
    def get_response(self, text: str, userid: str) -> str:
        session_id = f"chatgpt_{userid}"
        if text == "#清除":
            LLMSessionCache.delete(session_id)
            return "会话已清除"
        
        messages = LLMSessionCache.get(session_id, [])
        messages.append({"role": "user", "content": text})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                user=str(userid)
            )
            result = completion.choices[0].message.content
            if result:
                messages.append({"role": "assistant", "content": result})
                LLMSessionCache.set(session_id, messages)
            return result
        except Exception as e:
            logger.error(f"请求ChatGPT出现错误: {e}")
            return f"ChatGPT API 调用失败: {str(e)}"

    def get_media_name(self, filename: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": filename}
        ]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"} # 尝试使用JSON模式
            )
            return self._parse_media_name_from_response(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"ChatGPT辅助识别出错: {e}")
            return {"errorMsg": str(e)}

# --- Google Gemini 客户端 ---
class GeminiClient(LLMBaseClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get("gemini_key")
        if not api_key:
            raise ValueError("Gemini API密钥不能为空")
        
        self.model_name = config.get("gemini_model", "gemini-1.5-flash")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def get_response(self, text: str, userid: str) -> str:
        session_id = f"gemini_{userid}"
        if text == "#清除":
            LLMSessionCache.delete(session_id)
            return "会话已清除"
            
        chat_session = LLMSessionCache.get(session_id)
        if not chat_session:
            chat_session = self.model.start_chat(history=[])
            LLMSessionCache.set(session_id, chat_session)

        try:
            response = chat_session.send_message(text)
            return response.text
        except Exception as e:
            logger.error(f"请求Gemini出现错误: {e}")
            return f"Gemini API 调用失败: {str(e)}"

    def get_media_name(self, filename: str) -> Dict[str, Any]:
        prompt_parts = [self.prompt, filename]
        try:
            # Gemini 对 JSON 输出有更好的支持，直接请求
            response = self.model.generate_content(prompt_parts)
            return self._parse_media_name_from_response(response.text)
        except Exception as e:
            logger.error(f"Gemini辅助识别出错: {e}")
            return {"errorMsg": str(e)}

# --- Groq 客户端 ---
class GrokClient(LLMBaseClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get("grok_key")
        if not api_key:
            raise ValueError("Groq API密钥不能为空")

        self.model = config.get("grok_model", "llama3-70b-8192")
        self.client = Groq(api_key=api_key)

    def get_response(self, text: str, userid: str) -> str:
        session_id = f"grok_{userid}"
        if text == "#清除":
            LLMSessionCache.delete(session_id)
            return "会话已清除"
        
        messages = LLMSessionCache.get(session_id, [])
        messages.append({"role": "user", "content": text})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                user=str(userid)
            )
            result = completion.choices[0].message.content
            if result:
                messages.append({"role": "assistant", "content": result})
                LLMSessionCache.set(session_id, messages)
            return result
        except Exception as e:
            logger.error(f"请求Groq出现错误: {e}")
            return f"Groq API 调用失败: {str(e)}"

    def get_media_name(self, filename: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": filename}
        ]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            return self._parse_media_name_from_response(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq辅助识别出错: {e}")
            return {"errorMsg": str(e)}