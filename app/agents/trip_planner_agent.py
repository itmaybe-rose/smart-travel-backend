'''多智能体旅行规划系统'''
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool

from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel, Budget
from ..config import get_settings
from ._shared import build_planner_query, create_fallback_plan, enrich_images, parse_response


#========Agent提示词========
ATTRACTION_AGENT_PROMPT = """
  你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。
  **重要提示：**
  你必须使用工具来搜索景点！不要自己编造景点信息！

  **工具调用格式**
  使用maps_text_search工具时，必须严格按照以下格式：
  `[TOOL_CALL:amap_maps_text_search:keywords=景点关键词,city=城市名称]`

  **示例:**
  用户：“搜索北京的历史文化景点”
  你的回复：[TOOL_CALL:amap_maps_text_search:keywords=历史文化,city=北京]

  **注意:**
  1.必须使用工具,不要直接回答
  2.格式必须完全正确,包括方括号和冒号
  3.参数用逗号分割
"""

WEATHER_AGENT_PROMPT = """
  你是天气查询专家。你的任务是查询指定城市的天气。

  **重要提示：**
  你必须使用工具来查询天气！不要自己编造天气信息！

  **工具调用格式**
  使用maps_weather工具时，必须严格按照以下格式：
  `[TOOL_CALL:amap_maps_weather:city=城市名称]`

  **示例:**
  用户："查询北京的天气"
  你的回复：[TOOL_CALL:amap_maps_weather:city=北京]

  **注意:**
  1.必须使用工具,不要直接回答
  2.格式必须完全正确,包括方括号和冒号
"""

HOTEL_AGENT_PROMPT = """
  你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。
  **重要提示：**
  你必须使用工具来搜索酒店！不要自己编造酒店信息！

  **工具调用格式**
  使用maps_text_search工具时，必须严格按照以下格式：
  `[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=城市名称]`

  **示例:**
  用户："搜索北京的酒店"
  你的回复：[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=北京]

  **注意:**
  1.必须使用工具,不要直接回答
  2.格式必须完全正确,包括方括号和冒号
  3.关键词使用“酒店”或“宾馆”
"""

PLANNER_AGENT_PROMPT = """
  你是行程规划专家。你的任务是根据景点信息和天气信息，生成详细的旅行计划。
   
   请严格按照以下JSON格式返回旅行计划:
   ```json
   {
     "city": "城市名称",
     "start_date": "YYYY-MM-DD",
     "end_date": "YYYY-MM-DD",
     "days":[
      {
        "date": "YYYY-MM-DD",
        "day_index":0,
        "description":"第1天行程概述",
        "transportation":"交通方式",
        "accommodation":"住宿类型",
        "hotel":{
          "name":"酒店名称",
          "address":"酒店地址",
          "location":{"longitude":116.397128,"latitude":39.916527},
          "price_range":"300-500元",
          "estimated_cost":400
        },
        "attractions":[
          {
            "name":"景点名称",
            "address":"详细地址",
            "location":{"longitude":116.397128,"latitude":39.916527},
            "visit_duration": 120,
            "description": "景点详细描述",
            "category": "景点类别",
            "ticket_price": 60
          }
        ],
        "meals":[
          {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
          {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
          {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
        ]
      }
     ],
     "weather_info":[
        {
          "date": "YYYY-MM-DD",
          "day_weather":"晴",
          "night_weather":"多云",
          "day_temp":25,
          "night_temp":15
        }
     ],
     "overall_suggestion":"总体建议",
     "budget":{
        "total_attractions":180,
        "total_hotels":1200,
        "total_meals":480,
        "total_transportation":200,
        "total":2060
     }
   }
  
  **Prompt设计要点**
  1.**明确角色**:"你是XX专家"
  2.**强调工具使用**:"必须使用工具，不要编造"
  3.**提供格式示例**:避免格式错误
  4.**列出注意事项**:防止常见错误
"""

class MultiAgentTripPlanner:
  '''多智能体旅行规划系统'''

  def __init__(self):
    """初始化多智能体旅行规划系统"""
    print("♻️开始初始化多智能体旅行规划系统...")

    try:
      settings = get_settings()
      self.llm = get_llm()
      #步骤1:创建共享MCP工具
      '''MCP是一种基于Python的工具，用于创建和管理智能体之间的通信通道，
      可以在不同智能体之间传递信息和调用函数。
      标准化工具调用协议：
      传统方式：每个Agent都要写一遍API调用代码
      MCP方式：工具统一管理，Agent按需使用（创建一次，多个Agent使用）'''
      print(" - 创建共享MCP工具...")
      # uvx.exe在虚拟环境的Scripts目录下，用sys.executable推算绝对路径（子进程PATH不可靠）
      uvx_path = os.path.join(os.path.dirname(sys.executable), "uvx.exe")
      self.amap_tool = MCPTool(
        name="amap",
        description="高德地图服务",
        server_command=[
          uvx_path,
          "--with", "pydantic<2.11",  # amap-mcp-server的依赖与新版pydantic不兼容，强制用老版
          "amap-mcp-server"
        ],
        # 注意：MCP服务器要求的变量名是AMAP_MAPS_API_KEY；合并完整os.environ保证子进程有PATH等
        env={
          **os.environ,
          "AMAP_MAPS_API_KEY": settings.amap_api_key,
          "UV_DEFAULT_INDEX": "https://pypi.tuna.tsinghua.edu.cn/simple",
        },
        auto_expand=True # 自动展开工具调用格式
      )
      # 框架的register_tool要求工具带expandable=True属性才会展开为独立工具，
      # 但MCPTool类没有设置该属性（框架bug），手动补上
      self.amap_tool.expandable = True
      if not self.amap_tool._available_tools:
        print("⚠️ 警告: MCP服务器工具发现失败，amap工具将无法使用！")
      else:
        print(f"   ✅ MCP服务器就绪，发现{len(self.amap_tool._available_tools)}个原生工具")
      #步骤2：创建景点搜索Agent
      print(" - 创建景点搜索Agent...")
      self.attraction_agent = SimpleAgent(
        name = "景点搜索专家", #Agent名称，用于日志和调试
        llm = self.llm, # 使用共享的LLM模型
        system_prompt=ATTRACTION_AGENT_PROMPT # 系统提示，定义Agent的角色和任务（Agent的工作手册）
             )
      self.attraction_agent.add_tool(self.amap_tool)
      #步骤3：创建天气查询Agent
      print(" - 创建天气查询Agent...")
      self.weather_agent = SimpleAgent(
        name = "天气查询专家",
        llm = self.llm,
        system_prompt=WEATHER_AGENT_PROMPT
      )
      self.weather_agent.add_tool(self.amap_tool)
      #步骤4:创建酒店推荐Agent
      print(" - 创建酒店推荐Agent...")
      self.hotel_agent = SimpleAgent(
        name = "酒店推荐专家",
        llm = self.llm,
        system_prompt=HOTEL_AGENT_PROMPT
      )
      self.hotel_agent.add_tool(self.amap_tool)
      #步骤5：创建行程规划Agent(不需要工具)
      '''规划Agent的任务是整合信息+逻辑推理，不需要调用外部API：
      输入：景点信息，天气信息，酒店信息
      输出：结构化的旅行计划JSON
      '''
      print(" - 创建行程规划Agent...")
      self.planner_agent = SimpleAgent(
        name = "行程规划专家",
        llm = self.llm,
        system_prompt=PLANNER_AGENT_PROMPT
      )
      print("✅多智能体旅行规划系统初始化成功")
      print(f"  景点搜索Agent:{len(self.attraction_agent.list_tools())}个工具")
      print(f"  天气查询Agent:{len(self.weather_agent.list_tools())}个工具")
      print(f"  酒店推荐Agent:{len(self.hotel_agent.list_tools())}个工具")

    except Exception as e:
      print(f"❌初始化多智能体旅行规划系统失败: {str(e)}")
      import traceback
      traceback.print_exc()
      raise 

  def plan_trip(self,request:TripRequest)->TripPlan:
    '''
    使用多智能体协作生成旅行计划
    Args:
      request:旅行请求
    Returns:
      TripPlan:生成的旅行计划
    '''
    try:
      print(f"\n{'=' * 60}")
      print(f"🚀 开始多智能体协作规划旅行...")
      print(f"  目的地:{request.city}")
      print(f"日期:{request.start_date}至{request.end_date}")
      print(f"天数:{request.travel_days}天")
      print(f"偏好:{','.join(request.preferences) if request.preferences else '无'}")
      print(f"{'=' * 60}\n")

      attraction_response = ""
      weather_response = ""
      hotel_response = ""

      #步骤1-3：景点/天气/酒店三路并行（互相独立），单路失败自动重试后降级为空
      print("⚡ 步骤1-3: 并行检索景点、天气、酒店...")
      t0 = time.time()
      with ThreadPoolExecutor(max_workers=3) as pool:
        f_att = pool.submit(self._run_with_retry, self.attraction_agent, self._build_attraction_query(request))
        f_wea = pool.submit(self._run_with_retry, self.weather_agent, f"请查询{request.city}的天气")
        f_hot = pool.submit(self._run_with_retry, self.hotel_agent, f"请推荐{request.city}的{request.accommodation}酒店")
        attraction_response = f_att.result()
        weather_response = f_wea.result()
        hotel_response = f_hot.result()
      print(f"⚡ 三路并行完成，耗时{time.time() - t0:.1f}秒")
      if attraction_response: print(f"📍 景点搜索结果: {attraction_response[:200]}...")
      if weather_response: print(f"🌤 天气查询结果: {weather_response[:200]}...")
      if hotel_response: print(f"🏨 酒店推荐结果: {hotel_response[:200]}...")
      print()

      #步骤4：行程规划Agent整合信息生成计划
      print("📋 步骤4: 生成行程计划...")
      planner_query = build_planner_query(
        request,
        attraction_response,
        weather_response,
        hotel_response
      )
      planner_response = self.planner_agent.run(planner_query)
      print(f"行程规划结果: {planner_response[:300]}...\n") 

      #步骤5：解析最终计划（携带真实天气，供降级时复用）
      trip_plan = parse_response(planner_response,request,weather_response)
      #步骤5.5：为景点检索真实配图（Unsplash，失败静默跳过不阻塞）
      print("🖼 为景点检索配图...")
      enrich_images(trip_plan)
      print(f"{'=' * 60}")
      print(f"✅ 旅行计划生成完成！")
      print(f"{'=' * 60}\n")
      return trip_plan
    except Exception as e:
      print(f"❌生成旅行计划失败: {str(e)}")
      import traceback
      traceback.print_exc()
      return create_fallback_plan(request, weather_response)

  def _run_with_retry(self, agent:SimpleAgent, query:str, retries:int=1)->str:
    """执行单个Agent，失败自动重试（指数退避）；全部失败返回空串，由上层降级处理"""
    for attempt in range(retries + 1):
      try:
        return agent.run(query)
      except Exception as e:
        if attempt < retries:
          wait = 2 ** (attempt + 1)
          print(f"⚠️ {agent.name} 第{attempt + 1}次调用失败({str(e)[:80]})，{wait}秒后重试...")
          time.sleep(wait)
        else:
          print(f"❌ {agent.name} 重试后仍失败，该路数据将降级为空: {str(e)[:80]}")
    return ""

  def _build_attraction_query(self,request:TripRequest)->str:
    """构建景点搜索查询"""
    keywords = []
    if request.preferences:
      keywords = request.preferences[0] #取第一个偏好
    else:
      keywords ="景点"
    
    # 直接返回工具调用格式
    query = f"请使用amap_maps_text_search工具搜索{request.city}的{keywords}相关景点。\n[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]"
    return query

  # _build_planner_query / _condense / _enrich_images / _parse_response / _create_fallback_plan
  # 已统一收敛到 agents._shared 模块，避免两版重复实现

#全局多智能体系统实例（单例模式）
_multi_agent_planner = None

def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取全局多智能体系统实例（单例模式）"""
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner
 