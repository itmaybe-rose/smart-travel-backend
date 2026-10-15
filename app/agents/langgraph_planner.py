'''LangGraph 版多智能体旅行规划系统

用 StateGraph 重写原 HelloAgents 四步流水线：
  START ──┬─→ 景点检索 ─┐
          ├─→ 天气检索 ─┼─→ 行程规划 → 解析/降级
          └─→ 酒店检索 ─┘
检索三节点由 langgraph 原生并行调度（async 节点 fan-out/join），
节点内直接确定性调用高德 MCP 工具（不经过 LLM）：
  maps_text_search 取候选 POI → maps_geo 地理编码补真实坐标；
  maps_weather 直接返回结构化预报。
只有 plan 节点使用 LLM 做选点与行程编排。
对外接口与 HelloAgents 版完全一致：MultiAgentTripPlanner.plan_trip(request) -> TripPlan
'''
import ast
import asyncio
import json
import logging
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from ..config import get_settings
from ..models.schemas import Attraction, Budget, TransportSegment, TripPlan, TripRequest
from ._shared import build_planner_query, create_fallback_plan, enrich_images

logger = logging.getLogger(__name__)


#========Agent提示词========
# 检索环节已完全去 LLM 化（确定性工具直连），仅保留行程规划 Agent 的提示词。
# 输出格式由 with_structured_output(TripPlan) 的 JSON Schema 强制保证，
# 提示词只需描述任务与约束，不再手写 JSON 模板。
PLANNER_AGENT_PROMPT = """你是行程规划专家。请严格依据下方提供的真实候选数据为用户生成旅行计划。

硬性约束：
1. 景点与酒店只能从候选列表中选择，禁止编造候选之外的名称、地址、坐标；
2. 景点和酒店的经纬度坐标必须原样使用候选中的真实坐标，不得改写；
3. 每天安排 2-3 个景点，每天必须包含早、中、晚三餐；
4. 每天推荐一个具体酒店（从酒店候选中选）；
5. 结合候选的评分(rating)、营业时间(open_time)与景点间距安排游览顺序与交通方式；
6. 出行日期若超出天气预报覆盖范围，对应日期的天气可留空，不要用其他日期顶替；
7. meals.type 必须是 breakfast / lunch / dinner 三者之一。
"""


#========图状态定义========
class PlannerState(BaseModel):
  '''StateGraph共享状态（Pydantic 模型，支持不可序列化的 MCP session）'''
  model_config = ConfigDict(arbitrary_types_allowed=True)

  request: Dict[str, Any] = {}             # TripRequest.model_dump()，全程只读
  session: Optional[Any] = None            # 本次请求独占的MCP长连接（不可序列化）
  attraction_result: str = ""              # 景点候选JSON（含真实坐标）
  weather_result: str = ""                 # 天气预报JSON
  weather_raw: str = ""                    # 天气工具原始输出（降级复用）
  hotel_result: str = ""                   # 酒店候选JSON（含真实坐标）
  trip_plan: Optional[Any] = None          # planner 节点直接产出 TripPlan 实例


class MultiAgentTripPlanner:
  '''LangGraph 多智能体旅行规划系统（接口与HelloAgents版一致）'''

  # 候选数量按行程实际需要收紧（MCP server端串行处理约0.7s/次，调用数直接决定耗时）
  ATTRACTION_CANDIDATES = 8   # 2天行程选4-6个景点，8个候选有富余
  HOTEL_CANDIDATES = 8
  PER_KEYWORD_LIMIT = 3       # 每个偏好关键词各取前N条，保证美食/购物/景点多样性
  TOOL_CONCURRENCY = 5        # 同一长连接上工具调用并发上限，避免触发高德QPS限制

  # 地理编码结果进程内缓存：同一(城市,名称/地址)只编码一次，跨请求复用
  _geo_cache: Dict[Tuple[str, str], Any] = {}

  def __init__(self):
    """初始化：构建LLM → 连接MCP发现工具 → 编译StateGraph"""
    print("♻️ 开始初始化 LangGraph 多智能体旅行规划系统...")
    try:
      settings = get_settings()
      self._settings = settings

      # LLM：直接从环境变量构建，与HelloAgentsLLM读取的变量保持一致
      timeout = float(os.getenv("LLM_TIMEOUT", "600") or 600)
      self.llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "qwen3.8-max"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=timeout,
        max_retries=2,
        # 关闭千问思考链：规划环节输出大JSON，思考链会让耗时从分钟级降到十秒级。
        # 用 extra_body 而非 model_kwargs：enable_thinking 是 DashScope 扩展参数，
        # model_kwargs 会作为方法 kwarg 传入 parse() 触发 TypeError，extra_body 则合并进请求体。
        extra_body={"enable_thinking": False},
      )

      self._tool_map: Dict[str, Any] = {} # 工具名称 -> 工具实例
      self._graph = None # StateGraph 实例，编译后赋值

      # 结构化输出：用 JSON Schema 强制模型按 TripPlan 结构返回，
      # 省去手工切 JSON 与解析失败降级。注意：with_structured_output 返回新的
      # runnable，必须赋给独立变量，不能写回 self.llm（ChatOpenAI 是 Pydantic 实例，
      # 动态加属性会触发字段校验报错）。
      self.llm_structured = self.llm.with_structured_output(TripPlan, method="json_schema")

      # MCP连接与图编译需要异步；路由处于事件循环中不能直接asyncio.run，
      # 统一走线程桥接（与plan_trip同一套机制）
      self._bridge(self._init_mcp_and_graph())
      print("✅ LangGraph 多智能体旅行规划系统初始化成功")
    except Exception as e:
      logger.error("初始化 LangGraph 多智能体旅行规划系统失败: %s", str(e))
      import traceback
      traceback.print_exc()
      raise

  #--------初始化--------
  def _resolve_uvx(self) -> str:
    """定位uvx.exe：优先当前venv，没有则回退原venv（venv-lg不装uv工具链）"""
    candidates = [
      os.path.join(os.path.dirname(sys.executable), "uvx.exe"),
      # 本文件位于 backend/app/agents/，向上三级即backend根
      os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "venv", "Scripts", "uvx.exe"),
    ]
    for path in candidates:
      if os.path.exists(path):
        return path
    return "uvx"  # 都没有则交给PATH

  async def _init_mcp_and_graph(self):
    """连接高德MCP服务器、发现工具、绑定各检索Agent、编译StateGraph"""
    # 本地修复版 amap-mcp-server（原 PyPI 包 maps_direction_transit_integrated_by_coordinates
    # 里有 print(data) 把高德响应 dict repr 打到 stdout，stdio 协议通道被污染会导致 JSON-RPC 解析炸掉）
    local_server = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # .../backend/app/agents/
        "..", "..", "amap_mcp_server_local", "server.py",
    )
    local_server = os.path.normpath(local_server)

    python_exe = sys.executable  # 当前 venv 的 Python（venv-lg）
    self._client = MultiServerMCPClient({
      "amap": {
        "transport": "stdio",
        "command": python_exe,
        "args": [local_server, "stdio"],
        "env": {
          **os.environ,
          "AMAP_MAPS_API_KEY": self._settings.amap_api_key,
          "PYTHONIOENCODING": "utf-8",
          "PYTHONUTF8": "1",
        },
      }
    })
    tools = await self._client.get_tools()
    self._tool_map = {t.name: t for t in tools}
    if not self._tool_map:
      logger.warning("MCP服务器工具发现失败，检索Agent将无法使用！")
    else:
      print(f"   ✅ MCP服务器就绪，发现{len(self._tool_map)}个原生工具:")
      for name in sorted(self._tool_map.keys()):
        print(f"      - {name}")

    # 检索节点直接调用工具（不经LLM），只需确认必需工具齐备
    required = ("maps_text_search", "maps_weather", "maps_geo")
    missing = [n for n in required if n not in self._tool_map]
    if missing:
      logger.warning("MCP缺少工具%s，对应检索节点将降级为空结果", missing)

    self._build_graph()


  #Edge---用边声明拓扑
  def _build_graph(self):
    """编译StateGraph：START并行fan-out三路检索 → plan节点join → transport节点填交通 → END"""
    g = StateGraph(PlannerState) # 1.创建图，定义状态类型为 PlannerState
    #2.注册五个节点：景点、天气、酒店、计划、交通
    g.add_node("retrieve_attractions", self._node_attractions)
    g.add_node("retrieve_weather", self._node_weather)
    g.add_node("retrieve_hotels", self._node_hotels)
    g.add_node("plan", self._node_plan)
    g.add_node("transport", self._node_transport)
    # 3.同一节点(START)三条出边 → langgraph自动并发执行
    g.add_edge(START, "retrieve_attractions")
    g.add_edge(START, "retrieve_weather")
    g.add_edge(START, "retrieve_hotels")
    # 4.三路全部完成后才进入plan（天然join屏障）（三路都指向plan）
    g.add_edge("retrieve_attractions", "plan")
    g.add_edge("retrieve_weather", "plan")
    g.add_edge("retrieve_hotels", "plan")
    # 5.plan产出TripPlan → transport节点填真实交通段 → END
    g.add_edge("plan", "transport")
    g.add_edge("transport", END)
    #6.编译成可执行图
    self._graph = g.compile()

  #========图节点（检索节点：确定性工具直连，无LLM）========
  #一系列节点函数：景点、天气、酒店、计划
  async def _node_attractions(self, state: PlannerState) -> Dict[str, str]:
    req = state.request
    session = state.session
    city = req["city"]
    t0 = time.time()
    print("🔎 检索节点[景点]启动（maps_text_search 直连）...")
    try:
      # 每个偏好各搜一轮（配额取前N条保证多样性）+ 通用"景点"兜底（旧版只取第一个偏好）
      keywords: List[str] = list(dict.fromkeys(
        p for p in (req.get("preferences") or []) if p
      ))[:4]
      keywords.append("景点")
      kw_limits = [(kw, self.PER_KEYWORD_LIMIT) for kw in keywords[:-1]]
      kw_limits.append(("景点", self.ATTRACTION_CANDIDATES))  # 兜底词多取一些
      pois = await self._search_pois(session, kw_limits, city)
      candidates = await self._resolve_coords(
        session, pois[: self.ATTRACTION_CANDIDATES + 2], city, "景点")
      # 坐标距离去重：同一地点的子景点（如故宫博物院内的养心殿）只保留第一个
      candidates = self._dedup_by_distance(candidates, min_dist_m=200)
      candidates = candidates[: self.ATTRACTION_CANDIDATES]
      text = json.dumps({"pois": candidates}, ensure_ascii=False) if candidates else ""
      print(f"   ✅ [景点] {len(candidates)}个候选（含真实坐标），耗时{time.time() - t0:.1f}秒")
    except Exception as e:
      logger.error("[景点] 检索失败，该路降级为空: %s", str(e)[:100])
      text = ""
    return {"attraction_result": text}

  async def _node_weather(self, state: PlannerState) -> Dict[str, str]:
    req = state.request
    session = state.session
    city = req["city"]
    t0 = time.time()
    print("🔎 检索节点[天气]启动（maps_weather 直连）...")
    text, raw = "", ""
    try:
      data, raw = await self._call_tool_json(
        session, "maps_weather", {"city": city}, name="天气查询")
      forecasts = (data or {}).get("forecasts", [])
      if forecasts:
        text = json.dumps(
          {"city": (data or {}).get("city", city), "forecasts": forecasts},
          ensure_ascii=False)
        print(f"   ✅ [天气] {len(forecasts)}天预报，耗时{time.time() - t0:.1f}秒")
      else:
        logger.error("[天气] 未返回预报数据，该路降级为空")
    except Exception as e:
      logger.error("[天气] 查询失败，该路降级为空: %s", str(e)[:100])
    return {"weather_result": text, "weather_raw": raw}

  async def _node_hotels(self, state: PlannerState) -> Dict[str, str]:
    req = state.request
    session = state.session
    city = req["city"]
    t0 = time.time()
    print("🔎 检索节点[酒店]启动（maps_text_search 直连）...")
    try:
      # 只搜"酒店"：高德按热度权重排序，品牌酒店在前；"经济型酒店"这类词实测召回偏远小旅馆。
      # 住宿档次偏好已写入planner查询，由规划环节按rating/类型筛选
      pois = await self._search_pois(session, [("酒店", self.HOTEL_CANDIDATES + 4)], city)
      # 只保留住宿服务类POI（高德typecode 100000系列），剔除餐饮/购物等噪声
      pois = [p for p in pois if str(p.get("typecode", "")).startswith("10")] or pois
      candidates = await self._resolve_coords(
        session, pois[: self.HOTEL_CANDIDATES + 2], city, "酒店")
      candidates = candidates[: self.HOTEL_CANDIDATES]
      text = json.dumps({"pois": candidates}, ensure_ascii=False) if candidates else ""
      print(f"   ✅ [酒店] {len(candidates)}个候选（含真实坐标），耗时{time.time() - t0:.1f}秒")
    except Exception as e:
      logger.error("[酒店] 检索失败，该路降级为空: %s", str(e)[:100])
      text = ""
    return {"hotel_result": text}

  async def _node_plan(self, state: PlannerState) -> Dict[str, Any]:
    print("📋 步骤4: 生成行程计划（结构化输出）...")
    query = build_planner_query(
      state.request,
      state.attraction_result,
      state.weather_result,
      state.hotel_result,
    )
    # with_structured_output 直接返回 TripPlan 实例，无需手工解析 JSON
    trip_plan = await self.llm_structured.ainvoke([
      SystemMessage(content=PLANNER_AGENT_PROMPT),
      HumanMessage(content=query),
    ])
    return {"trip_plan": trip_plan}

  async def _node_transport(self, state: PlannerState) -> Dict[str, Any]:
    """交通规划节点：用高德 direction 工具为每天相邻景点生成真实路线段，
    填回 day.transport_segments 并重算预算。全程降级安全，绝不抛异常。"""
    print("🚗 交通规划节点启动...")
    trip_plan = state.trip_plan
    session = state.session
    req = state.request or {}
    city = req.get("city", "")
    mode = req.get("transportation", "公共交通") or "公共交通"
    if not trip_plan or not getattr(trip_plan, "days", None):
      print("   ⚠️ 无 trip_plan 或 days 为空，跳过交通规划")
      return {}

    # 解析要调用的方向工具
    tool_name, mode_label = self._pick_direction_tool(mode)
    sem = asyncio.Semaphore(self.TOOL_CONCURRENCY)

    async def _one_segment(fr: Attraction, to: Attraction) -> Optional[TransportSegment]:
      """调用一次 direction 工具，返回 TransportSegment 或 None"""
      try:
        fr_loc = fr.location
        to_loc = to.location
        if not fr_loc or not to_loc:
          return None
        fr_coord = f"{fr_loc.longitude},{fr_loc.latitude}"
        to_coord = f"{to_loc.longitude},{to_loc.latitude}"
        args: Dict[str, Any] = {"origin": fr_coord, "destination": to_coord}
        # 公交工具必须同时传 city 和 cityd（起点终点所在城市，跨城场景用）
        if "transit" in tool_name:
          args["city"] = city
          args["cityd"] = city
        async with sem:
          data, _ = await self._call_tool_json(
            session, tool_name, args,
            name=f"交通[{fr.name[:6]}→{to.name[:6]}]")
        parsed = self._parse_direction(data, tool_name)
        if not parsed:
          return None
        # 费用：API 优先，兜底估算
        api_cost = parsed.get("cost")
        km = parsed["distance"] / 1000.0
        seg_count = parsed.get("segments", 1)
        if api_cost is not None:
          cost_val = int(api_cost)
        else:
          cost_val = int(self._calc_transport_cost(mode, km, seg_count))
        return TransportSegment(
          from_name=fr.name,
          to_name=to.name,
          distance_km=round(km, 2),
          duration_min=max(1, round(parsed["duration"] / 60)),
          mode=mode_label,
          cost=cost_val,
          polyline=parsed.get("polyline", ""),
        )
      except Exception as e:
        logger.warning("交通段 %s→%s 生成失败: %s", fr.name, to.name, str(e)[:80])
        return None

    total_seg_count = 0
    total_transport_cost = 0
    try:
      for day in trip_plan.days:
        attractions = [a for a in (day.attractions or [])
                       if a.location and getattr(a.location, "longitude", None)]
        if len(attractions) < 2:
          day.transport_segments = []
          continue
        # 两两并行规划
        tasks = [_one_segment(attractions[i], attractions[i + 1])
                 for i in range(len(attractions) - 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        segments: List[TransportSegment] = []
        for r in results:
          if isinstance(r, Exception):
            logger.warning("交通段异常跳过: %s", str(r)[:80])
            continue
          if r is not None:
            segments.append(r)
            total_transport_cost += r.cost
        day.transport_segments = segments
        total_seg_count += len(segments)
    except Exception as e:
      logger.error("交通规划节点整体失败: %s", str(e)[:120])
      # 降级：所有 day 的 transport_segments 保持默认空列表

    # 重算预算
    try:
      budget = trip_plan.budget
      if budget is None:
        budget = Budget()
        trip_plan.budget = budget
      budget.total_transportation = total_transport_cost
      budget.total = (budget.total_attractions + budget.total_hotels
                      + budget.total_meals + total_transport_cost)
    except Exception as e:
      logger.warning("交通预算重算跳过: %s", str(e)[:80])

    # 步骤 10: 拼 route_map_url
    try:
      trip_plan.route_map_url = self._build_route_map_url(trip_plan)
    except Exception as e:
      logger.warning("route_map_url 生成跳过: %s", str(e)[:80])
      trip_plan.route_map_url = None

    _rmu = trip_plan.route_map_url
    print(f"✅ [交通] {total_seg_count}段路线，总费用¥{total_transport_cost}，"
          f"route_map_url={'有(' + str(len(_rmu)) + '字符)' if _rmu else '未生成'}")
    return {"trip_plan": trip_plan}

  #========确定性工具调用（无LLM，全部复用同一MCP长连接）========
  @staticmethod
  def _extract_tool_text(result: Any) -> str:
    """从 MCP CallToolResult 提取文本：适配 content块列表 / str / dict 三种形态"""
    if isinstance(result, str):
      return result
    content = getattr(result, "content", None)
    if content is not None:
      return "\n".join(
        getattr(c, "text", "") for c in content
        if getattr(c, "type", None) == "text"
      )
    if isinstance(result, (list, tuple)):
      parts = [
        b.get("text", "") for b in result
        if isinstance(b, dict) and b.get("type") == "text"
      ]
      return "\n".join(p for p in parts if p)
    if isinstance(result, dict):
      return result.get("text") or json.dumps(result, ensure_ascii=False)
    return str(result)

  @staticmethod
  def _valid_lnglat(lng: float, lat: float) -> bool:
    """粗校验中国经纬度范围，过滤异常值"""
    return 73.0 <= lng <= 136.0 and 3.0 <= lat <= 54.0

  @staticmethod
  def _safe_parse_json(text: str) -> Any:
    """先 json.loads，失败回退 ast.literal_eval（兼容单引号 Python repr）。"""
    try:
      return json.loads(text)
    except (json.JSONDecodeError, ValueError):
      return ast.literal_eval(text)

  async def _call_tool_json(self, session: Any, tool_name: str, args: Dict[str, Any],
                            name: str = "", retries: int = 1) -> Tuple[Any, str]:
    """在长连接session上调用高德MCP工具并解析JSON，失败指数退避重试一次。
    Returns: (解析后的dict/list, 原始JSON文本)；彻底失败返回 (None, "")
    """
    last_err = ""
    for attempt in range(retries + 1):
      try:
        result = await session.call_tool(tool_name, args)
        if getattr(result, "isError", False):
          raise RuntimeError(self._extract_tool_text(result)[:120])
        raw_text = self._extract_tool_text(result)
        return self._safe_parse_json(raw_text), raw_text
      except Exception as e:
        last_err = str(e)[:100]
        if attempt < retries:
          wait = 2 ** (attempt + 1)
          logger.warning("%s 第%d次调用失败(%s)，%d秒后重试...", name or tool_name, attempt + 1, last_err, wait)
          await asyncio.sleep(wait)
    logger.error("%s 重试后仍失败: %s", name or tool_name, last_err)
    return None, ""

  async def _search_pois(self, session: Any,
                         keyword_limits: List[Tuple[str, int]],
                         city: str = "") -> List[Dict[str, Any]]:
    """多关键词并发文本搜索，每个关键词只取前limit条（配额制保证多样性），
    按POI id去重并保持命中顺序；citylimit=true严格限定本市。
    注意：高德该接口返回的POI不含location，坐标需后续 maps_search_detail 补全。"""
    async def one(kw: str, limit: int) -> List[Dict[str, Any]]:
      data, _ = await self._call_tool_json(
        session, "maps_text_search",
        {"keywords": kw, "city": city, "citylimit": "true"},
        name=f"POI搜索[{kw}]")
      pois = (data or {}).get("pois", []) if isinstance(data, dict) else []
      return (pois or [])[:limit]

    results = await asyncio.gather(*(one(kw, lim) for kw, lim in keyword_limits))
    seen, merged = set(), []
    for pois in results:
      for p in pois:
        pid = p.get("id")
        if pid and pid not in seen:
          seen.add(pid)
          merged.append(p)
    return merged

  @classmethod
  def _parse_location_str(cls, loc: Any) -> Tuple[float, float]:
    """解析 "经度,纬度" 字符串，非法/越界返回None"""
    if not isinstance(loc, str) or "," not in loc:
      return None
    try:
      lng, lat = float(loc.split(",")[0]), float(loc.split(",")[1])
    except ValueError:
      return None
    return (lng, lat) if cls._valid_lnglat(lng, lat) else None

  @staticmethod
  def _parse_geo_location(data: Any) -> Tuple[float, float]:
    """解析 maps_geo 返回：location 形如 "121.477,31.234"。
    仅接受精确级别（兴趣点/门址/道路等），城市/区县中心点视为定位失败。"""
    weak_levels = {"城市", "区县", "省", "乡镇"}
    for item in (data or {}).get("return", []) if isinstance(data, dict) else []:
      if item.get("level") in weak_levels:
        continue
      loc = MultiAgentTripPlanner._parse_location_str(item.get("location"))
      if loc:
        return loc
    return None

  # ---------- 交通规划辅助函数 ----------

  @staticmethod
  def _pick_direction_tool(mode: str) -> Tuple[str, str]:
    """交通方式 → (MCP工具名, 模式描述)。"""
    m = (mode or "").strip()
    if m == "公共交通":
      return "maps_direction_transit_integrated_by_coordinates", "公交"
    if m in ("打车", "自驾"):
      return "maps_direction_driving_by_coordinates", m
    # 默认自驾兜底
    return "maps_direction_driving_by_coordinates", m or "自驾"

  @staticmethod
  def _parse_direction(data: Any, tool_name: str) -> Dict[str, Any]:
    """从 direction MCP 原始返回中抽取统一结构：
    {'distance': 米, 'duration': 秒, 'polyline': 'lng,lat;...', 'segments': N,
     'cost': float|None, 'tolls': float|None}
    cost: 高德 API 直接返回的费用（驾车 taxi_cost / cost.taxi），公交无此字段→None
    tolls: 过路费（仅驾车 show_fields=cost 时返回）
    调用方：优先用 cost；为 None 时再走 _calc_transport_cost 兜底估算。
    解析成功但无路线时返回 {}，并打 warning 日志（含起终点）。"""
    if not data or not isinstance(data, dict):
      return {}
    route = data.get("route") or {}

    if "transit" in tool_name:
      # ========== 公交 transit ==========
      # 高德 API 不返回公交票价，cost 保持 None，由调用方走 _calc_transport_cost 兜底
      transits = route.get("transits") or []
      if not transits:
        origin = route.get("origin") or ""
        destination = route.get("destination") or ""
        logger.warning("公交路线为空(%s)：origin=%s destination=%s，高德返回 status=%s",
                       tool_name, origin, destination, data.get("status"))
        return {}
      t = transits[0]
      distance = int(t.get("distance", 0) or 0)
      duration = int(t.get("duration", 0) or 0)
      seg_count = len(t.get("segments") or [])
      poly_parts: List[str] = []
      for seg in t.get("segments") or []:
        walking = seg.get("walking") or {}
        # walking.steps[].polyline（高德原始路径）
        for step in walking.get("steps") or []:
          pl = step.get("polyline")
          if pl:
            poly_parts.append(str(pl))
        # 兜底：walking 顶层 polyline（某些 MCP server 把它放这儿而非 steps 里）
        if not any((step.get("polyline") if isinstance(step, dict) else None)
                   for step in (walking.get("steps") or [])):
          pl = walking.get("polyline")
          if pl:
            poly_parts.append(str(pl))
        # 公交线路 polyline：兼容三种路径写法
        # 1. seg.buslines（原 PyPI 包可能扁平化）
        # 2. seg.bus_buslines（原 PyPI 包别名）
        # 3. seg.bus.buslines（高德原始嵌套 + 本地 server 结构）
        bus_list = []
        for key in ("buslines", "bus_buslines"):
          bus_list.extend(seg.get(key) or [])
        bus_obj = seg.get("bus") if isinstance(seg.get("bus"), dict) else {}
        bus_list.extend(bus_obj.get("buslines") or [])
        for bus in bus_list:
          pl = bus.get("polyline") if isinstance(bus, dict) else None
          if pl:
            poly_parts.append(str(pl))
      return {
        "distance": distance,
        "duration": duration,
        "polyline": ";".join(poly_parts),
        "segments": seg_count,
        "cost": None,      # 高德公交不返回票价
        "tolls": None,
      }
    else:
      # ========== 驾车 driving ==========
      paths = route.get("paths") or []
      if not paths:
        origin = route.get("origin") or ""
        destination = route.get("destination") or ""
        logger.warning("驾车路线为空(%s)：origin=%s destination=%s，高德返回 status=%s",
                       tool_name, origin, destination, data.get("status"))
        return {}
      p = paths[0]
      distance = int(p.get("distance", 0) or 0)
      duration = int(p.get("duration", 0) or 0)
      poly_parts: List[str] = []
      for step in p.get("steps") or []:
        pl = step.get("polyline")
        if pl:
          poly_parts.append(str(pl))
      if not poly_parts and p.get("path"):
        # 兜底：MCP server 可能剥掉 steps[].polyline 只保留顶层完整 path
        poly_parts.append(str(p["path"]))
      # 高德驾车费用提取（API 优先）
      # route 层级 taxi_cost: 预计出租车费用，最准，无需额外 show_fields
      taxi_cost = route.get("taxi_cost")
      if taxi_cost not in (None, "", "0", 0):
        api_cost = float(taxi_cost)
      else:
        # path 层级 cost.taxi（需 show_fields=cost 才返回）
        path_cost = p.get("cost") or {}
        api_taxi = path_cost.get("taxi")
        api_cost = float(api_taxi) if api_taxi not in (None, "", "0", 0) else None
      # 过路费（show_fields=cost 时返回）
      tolls_raw = (p.get("cost") or {}).get("tolls")
      api_tolls = float(tolls_raw) if tolls_raw not in (None, "", "0", 0) else None
      return {
        "distance": distance,
        "duration": duration,
        "polyline": ";".join(poly_parts),
        "segments": 1,
        "cost": api_cost,    # 高德估算的打车费用，自驾为 None 时由调用方兜底
        "tolls": api_tolls,  # 过路费（可选）
      }

  @staticmethod
  def _calc_transport_cost(mode: str, distance_km: float, segments: int = 1) -> float:
    """费用兜底估算——仅在 _parse_direction 返回 cost=None 时调用。
    高德 API 优先：
    - 驾车: route.taxi_cost / cost.taxi 直接返回，不走这里
    - 公交: API 根本不返回票价，始终走这里
    估算规则：
    - 公共交通: 3 元/段（按 transit 换乘段数）
    - 打车: 13 元起步（≤3km） + 2.5 元/km 超出部分
    - 自驾: 1 元/km（油费估算，不含过路费）
    """
    m = (mode or "").strip()
    if m == "公共交通":
      return round(3 * max(segments, 1), 2)
    if m == "打车":
      d = max(float(distance_km), 0.0)
      if d <= 3:
        return 13.0
      return round(13 + (d - 3) * 2.5, 2)
    if m == "自驾":
      return round(max(float(distance_km), 0.0) * 1.0, 2)
    return 0.0

  # ---------- 静态地图 URL 构建 ----------

  # 每天路径的折线颜色（最多 4 条 path，高德硬性限制）
  _DAY_PATH_COLORS = ["0xFF5722", "0x2196F3", "0x4CAF50", "0x9C27B0"]  # 红、蓝、绿、紫
  _DAY_MARKER_COLOR = "0xFF8C00"  # 景点标记：橙色
  _ROUTE_MAP_MARKER_MAX = 10      # 高德 markers 上限
  _ROUTE_MAP_PATH_MAX = 4         # 高德 paths 上限
  _ROUTE_MAP_POINTS_PER_PATH = 60   # 每条折线抽稀目标点数（750px 宽度足够，防 URL 超长）
  _ROUTE_MAP_COORD_DECIMALS = 5    # 坐标精度 5 位小数（≈1.1m，缩 URL 体积）
  _ROUTE_MAP_URL_MAX = 6000        # 代理转发给高德的最终 URL 长度上限（字符），超长逐级减点重拼

  @staticmethod
  def _thin_polyline(polyline: str, max_points: int) -> str:
    """高德 polyline 格式 'lng,lat;lng,lat;...' 按等间隔抽稀到 ≤ max_points 个点。"""
    if not polyline:
      return ""
    points = polyline.split(";")
    if len(points) <= max_points:
      return polyline
    # ceil 步进：刚好覆盖全部点
    step = -(-len(points) // max_points)
    return ";".join(points[::step])

  @staticmethod
  def _round_polyline(polyline: str, decimals: int) -> str:
    """坐标截断到指定小数位，缩短 URL 体积；跳过非法点。"""
    if not polyline:
      return ""
    out: List[str] = []
    for p in polyline.split(";"):
      xy = p.split(",")
      if len(xy) != 2:
        continue
      try:
        out.append(f"{round(float(xy[0]), decimals)},{round(float(xy[1]), decimals)}")
      except (TypeError, ValueError):
        continue
    return ";".join(out)

  @classmethod
  def _build_route_map_url(cls, trip_plan: TripPlan) -> Optional[str]:
    """把 TripPlan.days 拼成高德 staticmap 代理的相对 URL。

    返回形如 /api/trip/route-map?markers=...&paths=... 的同源路径，
    前端 <img :src> 直接用，html2canvas 导出 PDF 不会 tainted canvas。
    高德约束（这里主动消化）：
      - markers ≤ 10 个
      - paths ≤ 4 条
      - 每条折线抽稀到 ≤ 60 点、坐标 5 位小数，URL 超长自动逐级减点
    """
    days = getattr(trip_plan, "days", None)
    if not days:
      return None

    # --- 1. 收集所有景点坐标 → markers（按天顺序，最多 10 个） ---
    all_markers: List[str] = []
    for day in days:
      for a in (day.attractions or []):
        loc = a.location
        if loc and getattr(loc, "longitude", None):
          all_markers.append(f"{loc.longitude},{loc.latitude}")
    if not all_markers:
      return None
    all_markers = all_markers[:cls._ROUTE_MAP_MARKER_MAX]

    # --- 2. 收集每天 polyline 原文（最多 4 天） ---
    day_polylines: List[str] = []
    for day in days[:cls._ROUTE_MAP_PATH_MAX]:
      poly_parts = [s.polyline for s in (day.transport_segments or [])
                    if getattr(s, "polyline", None)]
      if poly_parts:
        day_polylines.append(";".join(poly_parts))

    # --- 3. 拼成 markers 串（每个 marker 独立一组 style，才能 A/B/C 标号） ---
    marker_parts: List[str] = []
    for i, coord in enumerate(all_markers):
      label = chr(ord('A') + i)  # A, B, C...（高德 marker label 支持 A-Z）
      marker_parts.append(f"mid,{cls._DAY_MARKER_COLOR},{label}:{coord}")
    markers_str = "|".join(marker_parts)

    # --- 4. 按点数预算抽稀拼 URL：超长时逐级减点重拼，直至 ≤ URL_MAX ---
    def make_paths(points: int) -> List[str]:
      out: List[str] = []
      for idx, poly in enumerate(day_polylines):
        sparse = cls._round_polyline(cls._thin_polyline(poly, points),
                                     cls._ROUTE_MAP_COORD_DECIMALS)
        color = cls._DAY_PATH_COLORS[idx % len(cls._DAY_PATH_COLORS)]
        # pathsStyle: weight,color,transparency,fillcolor,fillTransparency:location;...
        # 共 5 个字段（fillcolor/fillTransparency 留空 = 折线不封闭成多边形），
        # 实测 6 字段会被高德拒绝（HTTP 200 + JSON infocode 20003）
        out.append(f"4,{color},1,,:{sparse}")
      return out

    def assemble(paths_items: List[str]) -> str:
      # markers/paths 里的 : ; | , 都是高德保留符号，quote 时保留
      safe_chars = ":;,|"
      params = f"markers={urllib.parse.quote(markers_str, safe=safe_chars)}"
      if paths_items:
        paths_str = "|".join(paths_items)
        params += f"&paths={urllib.parse.quote(paths_str, safe=safe_chars)}"
      return f"/api/trip/route-map?{params}"

    points = cls._ROUTE_MAP_POINTS_PER_PATH
    url = assemble(make_paths(points))
    while len(url) > cls._ROUTE_MAP_URL_MAX and points > 10:
      points //= 2
      url = assemble(make_paths(points))
    return url

  # ---------- 地理编码 & 坐标解析 ----------

  async def _geocode(self, session: Any, query: str, city: str,
                     sem: asyncio.Semaphore) -> Tuple[float, float]:
    """名称/地址地理编码（带进程内缓存），失败返回None"""
    query = (query or "").strip()
    if not query:
      return None
    cache_key = (city, query)
    if cache_key in self._geo_cache:
      return self._geo_cache[cache_key]
    async with sem:
      data, _ = await self._call_tool_json(
        session, "maps_geo", {"address": query, "city": city},
        name=f"地理编码[{query[:12]}]")
    loc = self._parse_geo_location(data)
    self._geo_cache[cache_key] = loc
    return loc

  async def _poi_detail(self, session: Any, poi_id: str,
                        sem: asyncio.Semaphore) -> Dict[str, Any]:
    """用 maps_search_detail 取POI官方详情（含POI自身精确坐标），失败返回{}"""
    async with sem:
      data, _ = await self._call_tool_json(
        session, "maps_search_detail", {"id": poi_id},
        name=f"POI详情[{poi_id}]")
    return data if isinstance(data, dict) else {}

  @staticmethod
  def _dedup_by_distance(pois: List[Dict[str, Any]],
                         min_dist_m: float = 200) -> List[Dict[str, Any]]:
    """坐标距离去重：两个 POI 直线距离 < min_dist_m 米则视为同一地点，只保留第一个。
    典型场景：故宫博物院 vs 故宫博物院-养心殿（同址不同 POI ID）。"""
    import math
    kept: List[Dict[str, Any]] = []
    for p in pois:
      loc = p.get("location") or {}
      lng, lat = loc.get("longitude"), loc.get("latitude")
      if lng is None or lat is None:
        kept.append(p)  # 无坐标的不去重，留给后续过滤
        continue
      too_close = False
      for k in kept:
        kloc = k.get("location") or {}
        klng, klat = kloc.get("longitude"), kloc.get("latitude")
        if klng is None or klat is None:
          continue
        # Haversine 简化版（地球半径 6371000m）
        dlng, dlat = math.radians(klng - lng), math.radians(klat - lat)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat)) * math.cos(math.radians(klat))
             * math.sin(dlng / 2) ** 2)
        d = 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        if d < min_dist_m:
          too_close = True
          break
      if not too_close:
        kept.append(p)
    dropped = len(pois) - len(kept)
    if dropped:
      logger.info("坐标去重：%d个POI因距离<%dm被视为同址，已过滤", dropped, int(min_dist_m))
    return kept

  async def _resolve_coords(self, session: Any, pois: List[Dict[str, Any]],
                            city: str, label: str) -> List[Dict[str, Any]]:
    """并发为POI补真实坐标：
      1) maps_search_detail 取POI自身精确坐标（最准，附带评分/营业时间/中文分类）；
      2) 详情缺失时依次用地址、POI名称走 maps_geo 兜底；
      3) 都失败的POI直接过滤（schema中location必填，绝不允许编造坐标）。"""
    sem = asyncio.Semaphore(self.TOOL_CONCURRENCY)

    async def resolve_one(poi: Dict[str, Any]):
      loc = None
      detail = {}
      if poi.get("id"):
        detail = await self._poi_detail(session, poi["id"], sem)
        loc = self._parse_location_str(detail.get("location"))
      if loc is None:
        loc = await self._geocode(session, poi.get("address", ""), city, sem)
      if loc is None:
        loc = await self._geocode(session, poi.get("name", ""), city, sem)  # 地标名兜底
      if loc is None:
        return None
      return {
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "address": poi.get("address", ""),
        "typecode": poi.get("typecode", ""),
        "type": detail.get("type", ""),            # 中文分类，供planner判断category
        "rating": detail.get("rating", ""),        # 评分，辅助选点
        "open_time": detail.get("open_time", ""),  # 营业时间，辅助排程
        "location": {"longitude": loc[0], "latitude": loc[1]},
      }

    resolved = await asyncio.gather(*(resolve_one(p) for p in pois))
    candidates = [r for r in resolved if r]
    dropped = len(pois) - len(candidates)
    if dropped:
      logger.warning("[%s] %d个POI无法定位，已过滤（不编造坐标）", label, dropped)
    return candidates

  #========对外主入口========
  def plan_trip(self, request: TripRequest) -> TripPlan:
    """同步入口（供HelloAgents版保持接口一致 / 降级调用）。
    注意：内部会跨 loop 跑图，长任务不要在 FastAPI 事件循环里直接调，
    应使用 plan_trip_async 或外层 asyncio.to_thread 包装。"""
    return self._bridge(self._plan_trip_async(request))

  async def plan_trip_async(self, request: TripRequest) -> TripPlan:
    """异步主入口：FastAPI 路由直接 await，全程在同一事件循环内，
    MCP 长连接、图执行、配图都不再跨 loop（消除 _bridge 的线程+新loop开销）。"""
    return await self._plan_trip_async(request)

  async def _plan_trip_async(self, request: TripRequest) -> TripPlan:
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"🚀 [LangGraph] 开始多智能体协作规划旅行...")
    print(f"  目的地:{request.city}")
    print(f"日期:{request.start_date}至{request.end_date}")
    print(f"天数:{request.travel_days}天")
    print(f"偏好:{','.join(request.preferences) if request.preferences else '无'}")
    print(f"{'=' * 60}\n")

    weather_raw = ""
    try:
      # 图执行：整次请求复用一条MCP长连接（子进程只spawn一次），
      # 三路检索并发 + plan，全部在同一事件循环完成
      print("⚡ 步骤1-3: 并行检索景点、天气、酒店（StateGraph fan-out）...")
      raw_state = await self._run_graph(request)
      # LangGraph ainvoke 返回 dict（节点内部是 PlannerState 实例，出口序列化为dict），
      # 转回模型实例以便用属性访问
      state = raw_state if isinstance(raw_state, PlannerState) else PlannerState(**raw_state)
      print(f"⚡ 图执行完成，耗时{time.time() - t0:.1f}秒")
      if state.attraction_result: print(f"📍 景点候选: {state.attraction_result[:200]}...")
      if state.weather_result: print(f"🌤 天气预报: {state.weather_result[:200]}...")
      if state.hotel_result: print(f"🏨 酒店候选: {state.hotel_result[:200]}...")
      print()

      # planner 节点已通过 with_structured_output 直接产出 TripPlan 实例，无需解析
      trip_plan = state.trip_plan
      if trip_plan is None:
        raise RuntimeError("行程规划节点未返回结果")
      weather_raw = state.weather_raw
      print(f"行程规划完成：{trip_plan.city} {len(trip_plan.days)}天\n")
    except Exception as e:
      logger.error("生成旅行计划失败: %s", str(e))
      import traceback
      traceback.print_exc()
      trip_plan = create_fallback_plan(request, weather_raw)

    # 为景点检索真实配图（Unsplash，失败静默跳过不阻塞）
    print("🖼 为景点检索配图...")
    # fetch_attraction_image 是同步阻塞HTTP，丢线程池避免卡住事件循环
    await asyncio.to_thread(enrich_images, trip_plan)
    print(f"{'=' * 60}")
    print(f"✅ 旅行计划生成完成！总耗时{time.time() - t0:.1f}秒")
    print(f"{'=' * 60}\n")
    return trip_plan
  
  #========图执行========
  async def _run_graph(self, request: TripRequest) -> Dict[str, Any]:
    """建立本次请求独占的MCP长连接（uvx子进程只spawn一次），
    session放入state供三个检索节点复用，图执行完毕后自动关闭连接"""
    async with self._client.session("amap") as session:
      return await self._graph.ainvoke({
        "request": request.model_dump(), 
        "session": session,
      })

  def _bridge(self, coro):
    """同步世界 → 异步世界的桥：在独立线程里asyncio.run
    FastAPI路由运行在事件循环中，直接asyncio.run会报RuntimeError，
    新线程没有运行中的loop，可以安全启动。"""
    with ThreadPoolExecutor(max_workers=1) as pool:
      return pool.submit(asyncio.run, coro).result()

  # 解析/降级/配图/提示词构建 已统一收敛到 agents._shared 模块，避免两版重复实现

