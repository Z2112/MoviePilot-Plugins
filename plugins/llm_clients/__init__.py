from typing import Any, List, Dict, Tuple

from app.core.config import settings
from app.core.event import eventmanager, Event
from app.log import logger
from app.plugins import _PluginBase
# 修改引用，从新的 llm_clients 文件中导入工厂函数
from app.plugins.chatgpt.llm_clients import get_llm_client
from app.schemas.types import EventType, ChainEventType
from app.schemas import NotificationType

# 类名必须为 ChatGPT 以匹配目录名 chatgpt
class ChatGPT(_PluginBase):
    # 插件名称
    plugin_name = "LLM 服务"
    # 插件描述
    plugin_desc = "集成多种大语言模型（ChatGPT/Gemini/Groq）进行对话与媒体识别增强。"
    # 插件图标
    plugin_icon = "Chatgpt_A.png"
    # 插件版本
    plugin_version = "3.0.1"
    # 插件作者
    plugin_author = "jxxghp & Ht"
    # 作者主页
    author_url = "https://github.com/jxxghp"
    # 插件配置项ID前缀
    plugin_config_prefix = "llmservice_"
    # 加载顺序
    plugin_order = 15
    # 可使用的用户级别
    auth_level = 1

    # 私有属性
    llm_client = None  # 通用的 LLM 客户端实例
    _config = {}  # 存储完整配置
    _enabled = False
    _recognize = False
    _notify = False
    _llm_service = "chatgpt"

    # ChatGPT 专属的多 Key 管理属性
    _api_keys = []
    _current_key_index = 0
    _key_status = {}

    def init_plugin(self, config: dict = None):
        if not config:
            return

        self._config = config
        self._enabled = config.get("enabled")
        self._recognize = config.get("recognize")
        self._notify = config.get("notify")
        self._llm_service = config.get("llm_service", "chatgpt")

        if not self._enabled:
            return

        # 如果选择的是 ChatGPT，则初始化多 Key 支持
        if self._llm_service == 'chatgpt':
            openai_key = config.get("openai_key")
            if openai_key:
                self._api_keys = [key.strip() for key in openai_key.split(',') if key.strip()]
                self._key_status = {key: True for key in self._api_keys}
                logger.info(f"LLM 服务插件为 ChatGPT 加载了 {len(self._api_keys)} 个 API 密钥")
                if self._api_keys:
                    # 使用第一个密钥更新配置，以便初始化客户端
                    self._config["openai_key"] = self._api_keys[0]

        # 使用工厂函数获取并初始化对应的 LLM 客户端
        self.llm_client = get_llm_client(self._config)
        if self.llm_client:
            logger.info(f"LLM 服务插件已成功初始化，当前使用模型服务: {self._llm_service.upper()}")
        else:
            logger.error(f"LLM 服务插件初始化失败，请检查配置。")

    def switch_to_next_key(self, failed_key):
        """
        切换到下一个可用的API密钥（仅适用于ChatGPT）
        :return: (is_switched, error_message) 元组，表示是否切换成功及错误信息
        """
        if self._llm_service != 'chatgpt' or len(self._api_keys) <= 1:
            return False, "非 ChatGPT 服务或未配置多个密钥，无法切换。"

        self._key_status[failed_key] = False
        logger.warning(f"ChatGPT API 密钥 {failed_key[:8]}... 失效，尝试切换。")

        original_index = self._current_key_index
        for _ in range(len(self._api_keys)):
            self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
            next_key = self._api_keys[self._current_key_index]
            if self._key_status.get(next_key, True):
                break
        
        if all(not status for status in self._key_status.values()):
            logger.error("所有 ChatGPT API 密钥均已失效")
            return False, "所有 ChatGPT API 密钥均已失效，请检查配置"

        next_key = self._api_keys[self._current_key_index]
        logger.info(f"成功切换到下一个 ChatGPT API 密钥 {next_key[:8]}...")
        self._config["openai_key"] = next_key
        self.llm_client = get_llm_client(self._config)
        return self.llm_client is not None, ""

    def get_state(self) -> bool:
        return self._enabled

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        show_if_chatgpt = {'model': 'llm_service', 'operator': '==', 'value': 'chatgpt'}
        show_if_gemini = {'model': 'llm_service', 'operator': '==', 'value': 'gemini'}
        show_if_grok = {'model': 'llm_service', 'operator': '==', 'value': 'grok'}

        form_layout = [
            {
                'component': 'VForm',
                'content': [
                    {
                        'component': 'VRow',
                        'content': [
                            {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VSwitch', 'props': {'model': 'enabled', 'label': '启用插件'}}]},
                            {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VSwitch', 'props': {'model': 'recognize', 'label': '辅助识别'}}]},
                            {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VSwitch', 'props': {'model': 'notify', 'label': '开启通知'}}]},
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VSelect', 'props': {'model': 'llm_service', 'label': '选择语言模型服务', 'items': [{'title': 'ChatGPT', 'value': 'chatgpt'}, {'title': 'Google Gemini', 'value': 'gemini'}, {'title': 'Groq (Llama3等)', 'value': 'grok'}]}}]}
                        ]
                    },
                    {'component': 'VRow', 'condition': show_if_chatgpt, 'content': [
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VTextField', 'props': {'model': 'openai_url', 'label': 'OpenAI API Url', 'placeholder': 'https://api.openai.com'}}, {'component': 'VSwitch', 'props': {'model': 'proxy', 'label': '使用系统代理'}}, {'component': 'VSwitch', 'props': {'model': 'compatible', 'label': '兼容模式'}}]},
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VTextField', 'props': {'model': 'openai_key', 'label': 'API密钥 (多个以逗号分隔)', 'placeholder': 'sk-xxx,sk-yyy'}}]},
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 4}, 'content': [{'component': 'VTextField', 'props': {'model': 'model', 'label': '自定义模型', 'placeholder': 'gpt-4o'}}]}
                    ]},
                    {'component': 'VRow', 'condition': show_if_gemini, 'content': [
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 6}, 'content': [{'component': 'VTextField', 'props': {'model': 'gemini_key', 'label': 'Gemini API 密钥'}}]},
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 6}, 'content': [{'component': 'VTextField', 'props': {'model': 'gemini_model', 'label': 'Gemini 模型', 'placeholder': 'gemini-1.5-flash'}}]}
                    ]},
                    {'component': 'VRow', 'condition': show_if_grok, 'content': [
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 6}, 'content': [{'component': 'VTextField', 'props': {'model': 'grok_key', 'label': 'Groq API 密钥'}}]},
                        {'component': 'VCol', 'props': {'cols': 12, 'md': 6}, 'content': [{'component': 'VTextField', 'props': {'model': 'grok_model', 'label': 'Groq 模型', 'placeholder': 'llama3-70b-8192'}}]}
                    ]},
                    {'component': 'VRow', 'content': [
                        {'component': 'VCol', 'props': {'cols': 12}, 'content': [{'component': 'VTextarea', 'props': {'rows': 3, 'auto-grow': True, 'model': 'customize_prompt', 'label': '辅助识别提示词', 'hint': '所有模型进行辅助识别时共用的提示词'}}]}
                    ]},
                    {'component': 'VAlert', 'props': {'type': 'info', 'variant': 'tonal', 'text': '启用插件后，在消息交互时触发对话。开启辅助识别后，内置识别失败时将使用AI辅助。ChatGPT 支持输入多个API密钥（以英文逗号分隔），在密钥调用失败时将自动切换。'}}
                ]
            }
        ]
        default_data = {
            "enabled": False, "recognize": False, "notify": False, "llm_service": "chatgpt",
            "proxy": False, "compatible": False, "openai_url": "https://api.openai.com", "openai_key": "", "model": "gpt-4o",
            "gemini_key": "", "gemini_model": "gemini-1.5-flash", "grok_key": "", "grok_model": "llama3-70b-8192",
            "customize_prompt": '接下来我会给你一个电影或电视剧的文件名，你需要识别文件名中的名称、版本、分段、年份、分辨率、季、集等信息，并按以下JSON格式返回：{"name":string,"version":string,"part":string,"year":string,"resolution":string,"season":number|null,"episode":number|null}，特别注意返回结果需要严格附合JSON格式，不需要有任何其它的字符。如果中文文件名中存在谐音字或字母替代的情况，请还原最有可能的结果。'
        }
        return form_layout, default_data

    def _handle_api_error(self, response: Any, channel: str = None, userid: str = None) -> bool:
        is_error, error_msg = self.llm_client.is_api_error(response)
        if not is_error:
            return False

        if self._llm_service == 'chatgpt' and self._api_keys:
            current_key = self._api_keys[self._current_key_index]
            switched, switch_error = self.switch_to_next_key(current_key)
            if self._notify:
                message = f"ChatGPT API 密钥 {current_key[:8]}... 调用失败: {error_msg}"
                if channel: self.post_message(channel=channel, title=message, userid=userid)
                else: self.post_message(mtype=NotificationType.Plugin, title="LLM 服务", text=message)
                if not switched: self.post_message(mtype=NotificationType.Plugin, title="LLM 服务", text=switch_error)
            return not switched
        else:
            if self._notify:
                service_name = self._llm_service.upper()
                message = f"{service_name} API 调用失败: {error_msg}"
                if channel: self.post_message(channel=channel, title=message, userid=userid)
                else: self.post_message(mtype=NotificationType.Plugin, title="LLM 服务", text=message)
            return True

    @eventmanager.register(EventType.UserMessage)
    def talk(self, event: Event):
        if not self._enabled or not self.llm_client: return
        text = event.event_data.get("text")
        if not text or text.startswith(("http", "magnet", "ftp")): return
        userid, channel = event.event_data.get("userid"), event.event_data.get("channel")
        max_retries = len(self._api_keys) if self._llm_service == 'chatgpt' and self._api_keys else 1
        for _ in range(max_retries):
            response = self.llm_client.get_response(text=text, userid=userid)
            logger.info(f"LLM({self._llm_service}) 返回结果：{response}")
            is_final_error = self._handle_api_error(response, channel=channel, userid=userid)
            if is_final_error:
                if self._notify: self.post_message(channel=channel, title="无法获取LLM响应，请检查配置或API状态", userid=userid)
                return
            if not self.llm_client.is_api_error(response)[0]:
                self.post_message(channel=channel, title=response, userid=userid)
                return

    @eventmanager.register(ChainEventType.NameRecognize)
    def recognize(self, event: Event):
        if not self._recognize or not self.llm_client or not event.event_data: return
        title = event.event_data.get("title")
        if not title: return
        max_retries = len(self._api_keys) if self._llm_service == 'chatgpt' and self._api_keys else 1
        for _ in range(max_retries):
            response = self.llm_client.get_media_name(filename=title)
            logger.info(f"LLM({self._llm_service}) 辅助识别结果：{response}")
            is_error, _ = self.llm_client.is_api_error(response)
            if not is_error and isinstance(response, dict) and not response.get("name"): is_error = True
            if not is_error:
                event.event_data.update(response)
                return
            is_final_error = self._handle_api_error(response)
            if is_final_error:
                if self._notify: self.post_message(mtype=NotificationType.Plugin, title="LLM 服务", text=f"无法识别标题 {title}，API调用失败")
                return

    def stop_service(self):
        self.llm_client = None
