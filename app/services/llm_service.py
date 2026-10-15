'''LLM服务模块'''
from hello_agents import HelloAgentsLLM
from ..config import get_settings

#全局LLM实例
_llm_instance = None

def get_llm() -> HelloAgentsLLM:
  '''获取LLM实例(单例模式)  #全局只创建一个LLM实例，节省资源
  Returns:
    HelloAgentsLLM: LLM实例 
  '''
  global _llm_instance
  if _llm_instance is None:
    settings = get_settings()
    #HelloAgentsLLM会自动从环境变量中读取配置，无需手动传参
    _llm_instance = HelloAgentsLLM()
    print(f"✅LLM服务初始化成功")
    print(f"提供商：{_llm_instance.provider}")
    print(f"模型：{_llm_instance.model}")

  return _llm_instance

def reset_llm():
  '''重置LLM实例'''
  global _llm_instance
  _llm_instance = None
 