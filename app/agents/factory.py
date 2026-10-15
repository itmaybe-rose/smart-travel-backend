'''Agent实现切换工厂

通过环境变量AGENT_IMPL选择规划器实现：
  hello     - 原HelloAgents框架版（默认）
  langgraph - LangGraph StateGraph版
注意：uvicorn reload只监听.py文件，修改.env后需手动重启服务才能生效。
'''
import os
import threading

_instance = None
_lock = threading.Lock()


def get_trip_planner_agent():
  """获取全局规划器实例（单例模式，线程安全），接口与两个实现保持一致"""
  global _instance
  if _instance is not None:
    return _instance
  with _lock:
    if _instance is not None:
      return _instance
    # 必须先触发app.config导入（其模块级load_dotenv会加载.env），
    # 否则裸脚本/健康检查路径下AGENT_IMPL永远读不到
    from ..config import get_settings  # noqa: F401
    impl = os.getenv("AGENT_IMPL", "hello").strip().lower()
    if impl == "langgraph":
      try:
        from .langgraph_planner import MultiAgentTripPlanner
        _instance = MultiAgentTripPlanner()
        return _instance
      except ImportError as e:
        # 例如在只装了hello_agents的旧venv里选了langgraph：自动回退，避免服务崩溃
        print(f"⚠️ LangGraph实现不可用({e})，回退到HelloAgents实现")
    from .trip_planner_agent import MultiAgentTripPlanner
    _instance = MultiAgentTripPlanner()
  return _instance
