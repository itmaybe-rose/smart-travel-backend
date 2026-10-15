#配置管理模块
import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

#BaseSettings是Pydantic提供的基类，可以自动从环境变量读取
class Settings(BaseSettings):
  '''应用配置'''

  # Pydantic v2 用 model_config（旧版 class Config 小写会被忽略，导致 env_file/case_sensitive/extra 全部失效）
  model_config = SettingsConfigDict(
    env_file=".env",
    case_sensitive=False,
    extra="ignore",
  )

  #应用基本配置
  app_name :str = "HelloAgents智能旅行助手"
  app_version:str = "1.0.0"
  app_description:str = "基于HelloAgents框架的多智能体旅行规划系统"
  debug: bool = False

  #服务器配置
  host:str = "0.0.0.0"
  port:int = 8000

  #CORS配置
  cors_origins:str = "http://localhost:5173"

  #高德地图API配置
  amap_api_key:str = ""

  #Unsplash API配置
  unsplash_access_key:str = ""
  unsplash_secret_key:str = ""

  #->List[str] :获取CORS_origins列表
  def get_cors_origins_list(self) -> List[str]:
    return [origin.strip() for origin in self.cors_origins.split(",")]

#创建全局配置实例
settings = Settings()

def get_settings() -> Settings:
  return settings # 返回全局配置实例

def validate_config():
  '''验证配置是否完整'''
  errors = []
  if not settings.amap_api_key:
    errors.append("高德地图API密钥未配置")

  # LLM key 兼容多种命名：LLM_API_KEY（本项目统一）/ DASHSCOPE_API_KEY（千问）/ OPENAI_API_KEY（OpenAI兼容）
  llm_api_key = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
  )
  if not llm_api_key:
    errors.append("LLM_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY 未配置")

  if errors:
    error_msg = "配置错误:\n" + "\n".join(f"  -  {e}" for e in errors)
    raise ValueError(error_msg)

  return True #没有报错

#方便调试
def print_config():
  print(f"应用名称：{settings.app_name}")
  print(f"版本：{settings.app_version}")
  print(f"服务器：{settings.host}:{settings.port}")
  print(f"高德地图API Key:{'已配置' if settings.amap_api_key else '未配置'}")

  llm_api_key = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
  )
  llm_base_url = os.getenv("LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
  llm_model = os.getenv("LLM_MODEL_ID") or "qwen3.8-max"

  print(f"LLM API Key:{'已配置' if llm_api_key else '未配置'} ")
  print(f"LLM Base URL:{llm_base_url}")
  print(f"LLM Model:{llm_model}")