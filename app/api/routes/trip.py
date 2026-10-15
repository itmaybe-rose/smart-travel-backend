"""旅行规划API路由"""
import asyncio
import logging
import traceback
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ...config import get_settings
from ...models.schemas import TripPlanResponse, TripRequest
from ...agents.factory import get_trip_planner_agent

router = APIRouter(prefix="/trip", tags=["旅行规划"])
logger = logging.getLogger(__name__)


@router.post(
  "/plan",
  response_model=TripPlanResponse,
  summary="生成旅行计划",
  description="根据用户输入的旅行需求，生成对应的旅行计划。",
)
async def plan_trip(request: TripRequest):
  """生成旅行计划
  Args:
    request:旅行请求参数

  Returns:
    TripPlanResponse:生成的旅行计划
  """
  print(f"\n{'=' * 60}")
  print(f"📥 收到旅行规划请求:")
  print(f"   城市: {request.city}")
  print(f"   日期: {request.start_date} - {request.end_date}")
  print(f"   天数: {request.travel_days}")
  print(f"{'=' * 60}\n")
  agent = get_trip_planner_agent()
  try:
    # 优先走异步入口（LangGraph版：全程同一事件循环，不跨loop不阻塞）；
    # 没有异步入口的实现（HelloAgents版）退化为 to_thread 包装同步调用，
    # 同样不会卡住 FastAPI 事件循环。
    plan_async = getattr(agent, "plan_trip_async", None)
    if callable(plan_async):
      trip_plan = await plan_async(request)
    else:
      trip_plan = await asyncio.to_thread(agent.plan_trip, request)
    return TripPlanResponse(
      success=True,
      message="旅行计划生成成功",
      data=trip_plan,
    )
  except Exception as e:
    print(f"❌ 生成旅行计划失败: {str(e)}")
    traceback.print_exc()
    # 不把内部异常原文抛给客户端（可能含路径/key片段），统一文案，细节只进日志
    raise HTTPException(status_code=500, detail="生成旅行计划失败，请稍后重试")


@router.get(
  "/health",
  summary="健康检查",
  description="检查旅行规划服务是否正常",
)
async def health_check():
  """健康检查"""
  try:
    # 只校验实例是否成功创建，不触发耗时操作
    get_trip_planner_agent()
    return {
      "status": "healthy",
      "service": "trip_planner",
    }
  except Exception as e:
    raise HTTPException(
      status_code=503,
      detail=f"服务不可用: {str(e)}",
    )


# ---------- 步骤 9: 高德静态地图代理 ----------
# 必须代理的原因：PDF 导出用 html2canvas，跨域外链图会 tainted canvas 直接失败。

_AMAP_STATICMAP_URL = "https://restapi.amap.com/v3/staticmap"


@router.get(
  "/route-map",
  summary="高德静态地图代理",
  description="代理高德 staticmap 接口，前端同源访问避免跨域问题。"
              "markers/paths 由 transport 节点在后端拼好后透传。",
)
async def route_map(
  markers: str = Query(..., description="高德 markers 参数值，如 mid,0xFF8C00,A:116.4,39.9;..."),
  paths: str | None = Query(None, description="高德 paths 参数值，如 4,0xFF0000,1,,,:...|4,0x00FF00,...（最多4条）"),
  size: str = Query("750*400", description="地图尺寸，格式 W*H，最大 1024*1024"),
  scale: int = Query(2, description="1=普通图，2=高清图"),
) -> Response:
  """代理高德静态地图，返回 PNG 图片。

  高德 API 约束（由 transport 节点在拼 URL 时提前消化）：
    - markers 最多 10 个
    - paths 最多 4 条
    - size 最大 1024*1024
  """
  settings = get_settings()
  if not settings.amap_api_key:
    logger.error("route-map: amap_api_key 未配置")
    raise HTTPException(status_code=502, detail="地图服务未配置")

  # 上游 URL 的唯一事实来源：固定路径 + 组装的 params
  params: dict[str, Any] = {
    "key": settings.amap_api_key,
    "size": size,
    "scale": scale,
    "markers": markers,
  }
  if paths:
    params["paths"] = paths

  try:
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
      resp = await client.get(_AMAP_STATICMAP_URL, params=params)
  except httpx.RequestError as e:
    logger.error("route-map 代理请求失败: %s", str(e)[:120])
    raise HTTPException(status_code=502, detail="地图服务不可用")

  logger.info("route-map 上游 status=%s ct=%s bytes=%s",
              resp.status_code, resp.headers.get("content-type"), len(resp.content))

  # 高德 200 成功 → 透传图片
  if resp.status_code == 200:
    ctype = (resp.headers.get("content-type") or "").lower()
    # 高德出错时经常返回 HTTP 200 + JSON 错误体（如 INVALID_USER_KEY / 参数超限），
    # 此时必须按失败处理，否则 <img> 拿到 JSON 静默渲染失败
    if "json" in ctype or resp.content[:1] == b"{":
      logger.error("route-map 高德返回错误JSON(200): %s", resp.text[:200])
      raise HTTPException(status_code=502, detail="地图服务返回错误")
    return Response(content=resp.content, media_type=ctype or "image/png")

  # 高德返回了错误状态码：记录但统一对外 502，上游细节只进日志
  # 高德的非 200 响应通常是 JSON（status=0 + info）或图像占位图，不直接暴露给前端
  logger.error("route-map 上游异常: status=%s body=%s",
               resp.status_code, resp.text[:200] if "text" in (resp.headers.get("content-type") or "") else "<binary>")
  raise HTTPException(status_code=502, detail="地图服务暂时不可用")
