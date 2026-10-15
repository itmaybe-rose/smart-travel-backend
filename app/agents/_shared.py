"""两版 planner 共享的工具函数：提示词构建、文本压缩、降级计划、响应解析、配图。

原先 _build_planner_query / _condense / _enrich_images / _create_fallback_plan /
_parse_response 在 HelloAgents 版与 LangGraph 版中逐字重复，导致同一 bug（如降级计划
硬编码北京坐标）需改两处。这里统一收敛，两版只写各自的编排层。
"""
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

from ..models.schemas import (
  Attraction, Budget, DayPlan, Location, Meal, TripPlan, TripRequest, WeatherInfo,
)

logger = logging.getLogger(__name__)

# 主要城市中心经纬度，用于降级计划（无真实坐标时不硬编码北京）
CITY_CENTER: Dict[str, Dict[str, float]] = {
  "北京": {"longitude": 116.4074, "latitude": 39.9042},
  "上海": {"longitude": 121.4737, "latitude": 31.2304},
  "广州": {"longitude": 113.2644, "latitude": 23.1291},
  "深圳": {"longitude": 114.0579, "latitude": 22.5431},
  "杭州": {"longitude": 120.1551, "latitude": 30.2741},
  "苏州": {"longitude": 120.5853, "latitude": 31.2989},
  "南京": {"longitude": 118.7969, "latitude": 32.0603},
  "成都": {"longitude": 104.0668, "latitude": 30.5728},
  "重庆": {"longitude": 106.5516, "latitude": 29.5630},
  "西安": {"longitude": 108.9398, "latitude": 34.3416},
  "武汉": {"longitude": 114.3055, "latitude": 30.5928},
  "厦门": {"longitude": 118.0894, "latitude": 24.4798},
  "青岛": {"longitude": 120.3826, "latitude": 36.0671},
  "长沙": {"longitude": 112.9388, "latitude": 28.2282},
  "昆明": {"longitude": 102.8329, "latitude": 24.8801},
  "桂林": {"longitude": 110.2990, "latitude": 25.2742},
  "三亚": {"longitude": 109.5082, "latitude": 18.2479},
  "天津": {"longitude": 117.1901, "latitude": 39.1256},
  "洛阳": {"longitude": 112.4540, "latitude": 34.6197},
  "敦煌": {"longitude": 94.6623, "latitude": 40.1421},
  "丽江": {"longitude": 100.2330, "latitude": 26.8721},
  "拉萨": {"longitude": 91.1322, "latitude": 29.6600},
  "哈尔滨": {"longitude": 126.5358, "latitude": 45.8023},
  "大连": {"longitude": 121.6147, "latitude": 38.9140},
}


def condense(text: str, limit: int = 1200) -> str:
  """压缩上游返回文本，避免拉长思考模型的输入"""
  text = (text or "").strip()
  if len(text) <= limit:
    return text
  return text[:limit] + f"\n...(已截断，原文共{len(text)}字)"


def build_planner_query(request: Union[TripRequest, Dict[str, Any]],
                        attractions: str, weather: str, hotel: str = "") -> str:
  """构建行程规划查询。accept TripRequest 或 TripRequest.model_dump() 的 dict。"""
  if isinstance(request, dict):
    city = request["city"]
    days = request["travel_days"]
    start = request["start_date"]
    end = request["end_date"]
    transport = request["transportation"]
    acc = request["accommodation"]
    prefs = request.get("preferences") or []
    free = request.get("free_text_input") or ""
  else:
    city = request.city
    days = request.travel_days
    start = request.start_date
    end = request.end_date
    transport = request.transportation
    acc = request.accommodation
    prefs = request.preferences
    free = request.free_text_input or ""

  prefs_str = ", ".join(prefs) if prefs else "无"
  query = f"""请根据以下信息生成{city}的{days}天旅行计划：
    **基本信息**
    - 城市: {city}
    - 日期: {start} 至 {end}
    - 天数: {days}天
    - 交通方式: {transport}
    - 住宿: {acc}
    - 偏好: {prefs_str}

    **景点候选（高德地图实时检索结果，JSON格式）**
    字段说明：name名称, address地址, type中文分类, rating评分, open_time营业时间,
    location.longitude/latitude为高德返回的真实坐标
    {condense(attractions, 4000)}

    **天气预报（高德实时数据，JSON格式；仅未来4天，更远日期无实时预报）**
    {condense(weather, 1500)}

    **酒店候选（高德地图实时检索结果，JSON格式，坐标同为真实数据）**
    {condense(hotel, 3000)}

    **要求:**
    1. 每天安排2-3个景点，只能从上述候选中挑选，不要编造候选之外的景点
    2. 景点location必须原样使用候选中的真实坐标，禁止改写或臆造
    3. 每天必须包含早中晚三餐
    4. 每天推荐一个具体的酒店（从酒店候选中选）
    5. 结合rating评分、open_time营业时间和景点间距安排游览顺序与交通
    6. 出行日期超出天气预报范围时，weather_info对应日期可留空，不要拿其他日期顶替
    """
  if free:
    query += f"\n**额外要求**\n{free}"
  return query


def _extract_weather(weather_response: str, travel_days: int) -> List[WeatherInfo]:
  """从天气工具原始输出中提取真实预报（高德返回形如 {"date":..,"dayweather":..,"daytemp":..}）"""
  weather_info: List[WeatherInfo] = []
  if weather_response:
    for m in re.finditer(r'\{[^{}]*dayweather[^{}]*\}', weather_response):
      try:
        w = json.loads(m.group())
        weather_info.append(WeatherInfo(
          date=w.get("date", ""),
          day_weather=w.get("dayweather", ""),
          night_weather=w.get("nightweather", ""),
          day_temp=int(float(w.get("daytemp", 0))),
          night_temp=int(float(w.get("nighttemp", 0))),
        ))
      except Exception:
        continue
  return weather_info[:travel_days]


def create_fallback_plan(request: Union[TripRequest, Dict[str, Any]],
                         weather_response: str = "") -> TripPlan:
  """创建备用计划（当规划失败时）。尽量复用已获取的真实天气数据，
  坐标使用城市中心而非硬编码北京。"""
  if isinstance(request, dict):
    city = request["city"]
    days = request["travel_days"]
    start = request["start_date"]
    transport = request["transportation"]
    acc = request["accommodation"]
  else:
    city = request.city
    days = request.travel_days
    start = request.start_date
    transport = request.transportation
    acc = request.accommodation

  start_date = datetime.strptime(start, "%Y-%m-%d")
  center = CITY_CENTER.get(city, {"longitude": 116.4074, "latitude": 39.9042})

  weather_info = _extract_weather(weather_response, days)

  days_list = []
  for i in range(days):
    current_date = start_date + timedelta(days=i)
    days_list.append(DayPlan(
      date=current_date.strftime("%Y-%m-%d"),
      day_index=i,
      description=f"第{i + 1}天行程（降级生成，景点为占位）",
      transportation=transport,
      accommodation=acc,
      attractions=[
        Attraction(
          name=f"{city}景点{j + 1}",
          address=f"{city}市",
          location=Location(longitude=center["longitude"] + j * 0.005,
                            latitude=center["latitude"] + j * 0.005),
          visit_duration=120,
          description=f"这是{city}的著名景点",
          category="景点",
        )
        for j in range(2)
      ],
      meals=[
        Meal(type="breakfast", name=f"第{i + 1}天早餐", description="当地特色早餐"),
        Meal(type="lunch", name=f"第{i + 1}天午餐", description="午餐推荐"),
        Meal(type="dinner", name=f"第{i + 1}天晚餐", description="晚餐推荐"),
      ],
    ))

  weather_hint = ""
  if weather_info:
    w0 = weather_info[0]
    weather_hint = f"近期天气：{w0.day_weather}转{w0.night_weather}，{w0.day_temp}~{w0.night_temp}℃。"

  return TripPlan(
    city=city,
    start_date=start,
    end_date=(start_date + timedelta(days=days - 1)).strftime("%Y-%m-%d"),
    days=days_list,
    weather_info=weather_info,
    overall_suggestion=f"{weather_hint}本次为降级生成的行程，建议重新生成以获取更精准的安排。",
    budget=Budget(total=0),
  )


def parse_response(response: str, request: Union[TripRequest, Dict[str, Any]],
                   weather_response: str = "") -> TripPlan:
  """从 LLM 文本响应中提取 JSON 并解析为 TripPlan，失败时降级。
  （LangGraph 版已改用 with_structured_output，不再需要此函数；保留给 HelloAgents 版。）"""
  try:
    if "```json" in response:
      json_start = response.find("```json") + 7
      json_end = response.find("```", json_start)
      json_str = response[json_start:json_end].strip()
    elif "```" in response:
      json_start = response.find("```") + 3
      json_end = response.find("```", json_start)
      json_str = response[json_start:json_end].strip()
    elif "{" in response and "}" in response:
      json_start = response.find("{")
      json_end = response.rfind("}") + 1
      json_str = response[json_start:json_end]
    else:
      raise ValueError("响应中未找到JSON数据")

    data = json.loads(json_str)
    return TripPlan(**data)
  except Exception:
    return create_fallback_plan(request, weather_response)


def enrich_images(trip_plan: TripPlan) -> None:
  """为行程中所有景点并发补充配图（Unsplash），失败静默跳过。
  用 batch 并发替代逐景点串行，避免每个 3~5s 超时累加拖慢主流程。"""
  from ..services.image_service import fetch_attraction_images

  queries = []
  for day in trip_plan.days:
    for attraction in day.attractions:
      queries.append(f"{trip_plan.city} {attraction.name}")

  if not queries:
    return

  url_map = fetch_attraction_images(queries)
  hit = sum(1 for u in url_map.values() if u)
  logger.info("景点配图: %d/%d 命中", hit, len(queries))

  idx = 0
  for day in trip_plan.days:
    for attraction in day.attractions:
      attraction.image_url = url_map.get(queries[idx])
      idx += 1

