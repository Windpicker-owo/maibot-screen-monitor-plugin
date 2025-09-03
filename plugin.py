from typing import List, Tuple, Type, Optional, Dict
import asyncio
import datetime
import os
import base64
import pyautogui
from collections import deque
from openai import AsyncOpenAI
from src.common.logger import get_logger
from src.plugin_system import (
    BasePlugin, BaseEventHandler, BaseTool, register_plugin,
    ComponentInfo, ConfigField, ToolParamType, EventType,
    tool_api
)

logger = get_logger("screen_monitor_plugin")


    
class ScreenRecordStorage:
    """屏幕记录存储类 - 单例模式，负责统一存储和读取屏幕监控记录"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.results = deque(maxlen=1000)
        return cls._instance
    
    def add_record(self, record: dict):
        """添加记录"""
        self.results.append(record)
    
    def get_records(self, duration_minutes: int = 30) -> List[Dict]:
        """获取指定时间内的记录"""
        import datetime
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(minutes=duration_minutes)
        
        history = []
        for result in self.results:
            if isinstance(result, dict) and 'timestamp' in result:
                try:
                    timestamp = datetime.datetime.fromisoformat(result['timestamp'])
                    if timestamp > cutoff:
                        history.append(result)
                except (ValueError, TypeError):
                    continue
        return history
    
    def clear_old_records(self, retention_minutes: int):
        """清理超过保留时间的记录"""
        import datetime
        if not self.results:
            return
            
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(minutes=retention_minutes)
        
        # 从开头开始删除过期的记录
        while self.results and datetime.datetime.fromisoformat(self.results[0]['timestamp']) < cutoff:
            self.results.popleft()

class ScreenMonitor:
    def __init__(self, save_path: str, retention_minutes: int = 5, vlm_config: dict = None, llm_config: dict = None):
        self.save_path = save_path
        self.retention_minutes = retention_minutes
        self.vlm_config = vlm_config or {}
        self.llm_config = llm_config or {}
        self.record_storage = ScreenRecordStorage()  # 使用统一的记录存储
        self.running = False
        self.task = None
        self.temp_image_path = os.path.join(save_path, "latest_screen.png")
        self._ensure_save_dir()
        
        # 初始化OpenAI客户端
        self.vlm_client = AsyncOpenAI(
            base_url=self.vlm_config.get("base_url", "http://localhost:8000/v1"),
            api_key=self.vlm_config.get("api_key", "dummy")
        )
        self.llm_client = AsyncOpenAI(
            base_url=self.llm_config.get("base_url", "http://localhost:8000/v1"),
            api_key=self.llm_config.get("api_key", "dummy")
        )

    def _ensure_save_dir(self):
        """确保保存目录存在"""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    async def analyze_image(self, image_path: str) -> str:
        """使用VLM模型分析图片"""
        try:
            with open(image_path, "rb") as image_file:
                response = await self.vlm_client.chat.completions.create(
                    model=self.vlm_config.get("model", "gpt-4-vision"),
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的图像分析助手，请描述图片中的内容，特别关注用户的操作行为。"
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "请辨别出屏幕上窗口的主次，全面的描述主窗口的的内容（包括细节）和活动窗口标题。同时你需要简略的描述屏幕上的其他部分，注意，不要使用markdown格式！只输出纯文本即可！"
                                }
                            ]
                        }
                    ],
                )
            result =  response.choices[0].message.content
            logger.info(f"麦麦偷偷瞄了一眼你的屏幕，并看到了: {result[:50]}{'...' if len(result) > 50 else ''}")
            return result
        except Exception as e:
            logger.error(f"图像分析失败: {str(e)}")
            return "图像分析失败"

    async def summarize_activities(self, activities: List[str], duration: int) -> str:
        """使用LLM模型总结活动"""
        try:
            activities_text = "\n".join(activities)
            response = await self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的行为分析助手，请总结用户的屏幕活动。"
                    },
                    {
                        "role": "user",
                        "content": f"请总结过去{duration}分钟内的以下屏幕活动：\n\n{activities_text}，控制在1000字左右"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"活动总结失败: {str(e)}")
            return "活动总结失败"

    async def start_monitoring(self, interval: int):
        """开始监控"""
        self.running = True
        logger.info(f"屏幕监控开始运行，间隔: {interval}秒")
        iteration = 0
             
        while self.running:
            try:
                iteration += 1
                logger.debug(f"屏幕监控第 {iteration} 次循环开始")

                # 检查是否被取消
                await asyncio.sleep(0)  # 让出控制权，检查取消状态
                
                # 截图
                screenshot = pyautogui.screenshot()
                timestamp = datetime.datetime.now()
                logger.debug(f"截图完成: {timestamp}")
                
                # 保存为最新的临时文件
                screenshot.save(self.temp_image_path)
                logger.debug("截图保存完成")
                
                # VLM识别
                result = await self.analyze_image(self.temp_image_path)
          
                # 保存结果
                record = {
                    'timestamp': timestamp.isoformat(),
                    'description': result
                }
                self.record_storage.add_record(record)
                logger.debug(f"结果保存完成，当前记录数: {len(self.record_storage.results)}")
                
                # 清理超过保留时间的记录
                self.record_storage.clear_old_records(self.retention_minutes)
                logger.debug(f"清理后记录数: {len(self.record_storage.results)}")
                
                # 等待下一次截图
                logger.debug(f"等待 {interval} 秒后继续...")
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("屏幕监控任务收到取消信号，正在停止...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"监控出错: {str(e)}")
                if self.running:  # 只有在仍在运行时才继续
                    await asyncio.sleep(5)  # 出错后等待一段时间再继续

    def _cleanup_old_records(self):
        """清理超过保留时间的记录"""
        self.record_storage.clear_old_records(self.retention_minutes)

    async def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                # 等待任务完成，设置超时
                await asyncio.wait_for(self.task, timeout=5.0)
                logger.info("屏幕监控任务已正常停止")
            except asyncio.TimeoutError:
                logger.warning("屏幕监控任务停止超时，强制取消")
            except asyncio.CancelledError:
                logger.info("屏幕监控任务已被取消")
            except Exception as e:
                logger.error(f"停止屏幕监控任务时发生异常: {e}")
            finally:
                self.task = None

    def get_history(self, duration_minutes: int = 30) -> List[Dict]:
        """获取指定时间内的历史记录"""
        logger.debug(f"获取历史记录: 持续时间={duration_minutes}分钟")
        history = self.record_storage.get_records(duration_minutes)
        logger.debug(f"最终获取到的历史记录数量: {len(history)}")
        return history
    

class ScreenHistoryTool(BaseTool):
    """屏幕活动历史工具"""

    name = "screen_monitor"
    description = "获取屏幕历史活动的接口工具，不对llm开放"
    parameters = [("arg", ToolParamType.STRING, "占位参数，无作用", True, None)]
    available_for_llm = False

    @property
    def vlm_config(self):
        return {
            "base_url": self.get_config("vlm.base_url"),
            "api_key": self.get_config("vlm.api_key"),
            "model": self.get_config("vlm.model"),
        }

    @property
    def llm_config(self):
        return {
            "base_url": self.get_config("llm.base_url"),
            "api_key": self.get_config("llm.api_key"),
            "model": self.get_config("llm.model"),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._screen_monitor_instance = None
    
    @property
    def screen_monitor(self) -> ScreenMonitor:
        if self._screen_monitor_instance is None:
            # 获取插件根目录作为保存路径
            save_path = os.path.dirname(os.path.abspath(__file__))

            self._screen_monitor_instance = ScreenMonitor(
                save_path=save_path,
                retention_minutes=self.get_config("monitor.retention_minutes"),
                vlm_config=self.vlm_config,
                llm_config=self.llm_config
            )
        return self._screen_monitor_instance
    
    def initialize(self):
        # 启动监控任务
        interval = self.get_config("monitor.interval")
        logger.info(f"启动屏幕监控，间隔: {interval}秒")
        self.screen_monitor.task = asyncio.create_task(
            self.screen_monitor.start_monitoring(interval)
        )
        logger.info(f"屏幕监控插件已初始化，任务ID: {id(self.screen_monitor.task)}")
    
    async def execute(self, _) -> None:
        pass

class ScreenMonitorLauchHandler(BaseEventHandler):
    """屏幕监控事件处理器"""

    event_type = EventType.ON_START
    handler_name = "screen_monitor_laucher"
    handler_description = "自启动屏幕监控循环"
    weight = 0
    intercept_message = False

    async def execute(self, _) -> Tuple[bool, bool, Optional[str]]:      
        try:
            monitor_tool = tool_api.get_tool_instance("screen_monitor")
            if not monitor_tool:
                logger.error("屏幕监控工具未找到，请确保插件已正确加载。")
                return False, False, "屏幕监控工具未找到"
            
            # 确保屏幕监控已初始化
            retention_minutes = self.get_config("monitor.retention_minutes",5)
            username = self.get_config('monitor.username','')
            if not monitor_tool.screen_monitor.task:
                logger.info("屏幕监控未启动，正在初始化...")
                monitor_tool.initialize()
            else:
                logger.info(f"屏幕监控任务状态: 运行中={monitor_tool.screen_monitor.running}, 任务={monitor_tool.screen_monitor.task}")
                # 检查是否有历史记录
                history_count = len(monitor_tool.screen_monitor.record_storage.results)
                logger.info(f"当前屏幕监控记录数量: {history_count}")
            
            # 方法注入逻辑
            logger.info("开始注入 build_prompt_reply_context 方法...")
            from src.chat.replyer.default_generator import DefaultReplyer
            
            # 保存原始方法引用（只保存一次）
            if not hasattr(DefaultReplyer, '_original_build_prompt_reply_context'):
                DefaultReplyer._original_build_prompt_reply_context = DefaultReplyer.build_prompt_reply_context
            
            # 创建包装方法
            async def wrapped_build_prompt_reply_context(self, *args, **kwargs):
                """
                包装函数：增强build_prompt_reply_context方法，添加屏幕历史记录获取功能
                """
                # 1. 调用原始方法获取prompt和表达式
                prompt, selected_expressions = await DefaultReplyer._original_build_prompt_reply_context(self, *args, **kwargs)
                
                try:
                    # 2. 尝试获取屏幕监控工具实例
                    current_monitor_tool = tool_api.get_tool_instance("screen_monitor")
                    if not current_monitor_tool:
                        logger.debug("屏幕监控工具未找到，返回原始prompt")
                        return prompt, selected_expressions
                    
                    # 3. 获取屏幕历史记录（带超时机制）
                    try:
                        screen_history = current_monitor_tool.screen_monitor.get_history(duration_minutes=retention_minutes)
                        logger.debug(f"获取到屏幕历史记录: {len(screen_history)} 条记录")
                        if not screen_history:
                            logger.debug("屏幕历史记录为空，可能监控尚未开始或没有活动记录")
                            return prompt, selected_expressions
                    except asyncio.TimeoutError:
                        logger.warning("获取屏幕历史记录超时")
                        return prompt, selected_expressions
                    except Exception as e:
                        logger.error(f"获取屏幕历史记录失败: {e}")
                        return prompt, selected_expressions
                    
                    # 4. 格式化历史记录并插入到prompt中
                    if screen_history and isinstance(screen_history, list):
                        history_text = f"用户{username}屏幕活动历史记录：\n"
                        # 使用LLM总结活动
                        try:
                            summary = await current_monitor_tool.screen_monitor.summarize_activities(
                                [f"[{record['timestamp']}] {record['description']}" for record in screen_history],
                                duration=retention_minutes
                            )
                            history_text += summary
                        except Exception as e:
                            logger.error(f"屏幕活动总结失败: {e}")
                            # 如果总结失败，直接使用原始记录
                            for record in screen_history[-5:]:  # 只取最近5条记录
                                timestamp = record.get('timestamp', '')
                                description = record.get('description', '')
                                # 格式化时间戳
                                if timestamp:
                                    try:
                                        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                        formatted_time = dt.strftime('%H:%M:%S')
                                        history_text += f"- {formatted_time}: {description}\n"
                                    except ValueError:
                                        history_text += f"- {timestamp}: {description}\n"
                                else:
                                    history_text += f"- 未知时间: {description}\n"
                        # 在开头插入历史记录
                        prompt = f"{history_text}\n\n{prompt}"
                    
                    return prompt, selected_expressions
                    
                except Exception as e:
                    logger.error(f"屏幕历史记录注入失败: {e}")
                    # 确保错误不影响正常流程
                    return prompt, selected_expressions
            
            # 替换方法
            DefaultReplyer.build_prompt_reply_context = wrapped_build_prompt_reply_context
            
            logger.info("build_prompt_reply_context 方法注入成功")
            return True, True, "屏幕监控已启动且方法注入成功"
            
        except Exception as e:
            logger.error(f"启动屏幕监控和方法注入失败: {str(e)}")
            return False, False, f"启动屏幕监控失败: {str(e)}"

class ScreenMonitorStopHandler(BaseEventHandler):
    """屏幕监控停止事件处理器"""

    event_type = EventType.ON_STOP
    handler_name = "screen_monitor_stop_handler"
    handler_description = "停止屏幕监控并清理资源"
    weight = 0
    intercept_message = False

    async def execute(self, _) -> Tuple[bool, bool, Optional[str]]:
        try:
            monitor_tool = tool_api.get_tool_instance("screen_monitor")
            if not monitor_tool:
                logger.warning("屏幕监控工具未找到，无法停止监控。")
                return True, True, "屏幕监控工具未找到"

            # 停止监控
            await monitor_tool.screen_monitor.stop_monitoring()
            logger.info("屏幕监控已成功停止")
            return True, True, "屏幕监控已成功停止"
        except Exception as e:
            logger.error(f"停止屏幕监控失败: {str(e)}")
            return False, False, f"停止屏幕监控失败: {str(e)}"
    
@register_plugin
class ScreenMonitorPlugin(BasePlugin): 
    """屏幕监控插件 - 监控并总结屏幕活动"""
    
    plugin_name = "screen_monitor_plugin"
    enable_plugin = True
    dependencies = []
    python_dependencies = ["pyautogui", "openai"]
    config_file_name = "config.toml"
    
    config_schema = {
        "plugin": {
            "name": ConfigField(
                type=str, default="screen_monitor_plugin", description="插件名称"
            ),
            "version": ConfigField(type=str, default="1.0.0", description="插件版本"),
            "config_version": ConfigField(type=str, default="1.0.2", description="配置文件版本"),
            "enabled": ConfigField(
                type=bool, default=False, description="是否启用插件"
            ),
        },
        "monitor": {
            "username": ConfigField(
                type=str,
                default="default_user",
                description="监控用户名，用于标识记录"
            ),
            "interval": ConfigField(
                type=int,
                default=60,
                description="截图间隔（秒）"
            ),
            "retention_minutes": ConfigField(
                type=int,
                default=5,
                description="历史记录保留时间（分钟）"
            ),
        },
        "vlm": {  # 视觉语言模型配置
            "base_url": ConfigField(
                type=str,
                default="http://localhost:8000/v1",
                description="VLM API基础URL"
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="VLM API密钥"
            ),
            "model": ConfigField(
                type=str,
                default="gpt-4-vision",
                description="使用的VLM模型标识符"
            )
        },
        "llm": {  # 语言模型配置
            "base_url": ConfigField(
                type=str,
                default="http://localhost:8000/v1",
                description="LLM API基础URL"
            ),
            "api_key": ConfigField(
                type=str,
                default="",
                description="LLM API密钥"
            ),
            "model": ConfigField(
                type=str,
                default="gpt-4",
                description="使用的LLM模型标识符"
            )
        }
    }
    
    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        """返回插件组件"""
        return [
            (ScreenHistoryTool.get_tool_info(), ScreenHistoryTool),
            (ScreenMonitorLauchHandler.get_handler_info(), ScreenMonitorLauchHandler),
            (ScreenMonitorStopHandler.get_handler_info(), ScreenMonitorStopHandler)
        ]

