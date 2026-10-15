"""景点配图服务：从Unsplash按关键词检索真实照片

Unsplash图库以英文内容为主，中文关键词往往返回0条结果，
因此先用原始中文查询，空结果时自动转换为英文重试。

注意：Unsplash API 在国内网络环境下不稳定（偶发 ConnectTimeout / 403 限流），
所有调用必须短超时 + 静默失败，绝不阻塞主流程。
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# 进程内缓存：同一关键词只查一次（带锁保证线程安全）
_cache: Dict[str, Optional[str]] = {}
_cache_lock = threading.Lock()

# 常见城市中英映射
CITY_MAP = {
  "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou", "深圳": "Shenzhen",
  "杭州": "Hangzhou", "苏州": "Suzhou", "南京": "Nanjing", "成都": "Chengdu",
  "重庆": "Chongqing", "西安": "Xi'an", "武汉": "Wuhan", "厦门": "Xiamen",
  "青岛": "Qingdao", "长沙": "Changsha", "昆明": "Kunming", "桂林": "Guilin",
  "三亚": "Sanya", "天津": "Tianjin", "洛阳": "Luoyang", "敦煌": "Dunhuang",
  "丽江": "Lijiang", "拉萨": "Lhasa", "哈尔滨": "Harbin", "大连": "Dalian",
}

# 地标/类别关键词映射
LANDMARK_MAP = [
  ("故宫", "Forbidden City"), ("天安门", "Tiananmen"), ("长城", "Great Wall"),
  ("颐和园", "Summer Palace"), ("天坛", "Temple of Heaven"), ("圆明园", "Old Summer Palace"),
  ("恭王府", "Prince Gong Mansion"), ("雍和宫", "Yonghe Temple"), ("北海公园", "Beihai Park"),
  ("什刹海", "Shichahai"), ("后海", "Houhai Lake"), ("南锣鼓巷", "Nanluoguxiang hutong"),
  ("外滩", "the Bund"), ("兵马俑", "Terracotta Warriors"), ("西湖", "West Lake"),
  ("博物馆", "museum"), ("公园", "park"), ("寺庙", "temple"), ("寺", "temple"),
  ("庙", "temple"), ("宫殿", "palace"), ("宫", "palace"), ("湖", "lake"),
  ("山", "mountain"), ("塔", "tower"), ("江", "river"), ("河", "river"),
  ("海滩", "beach"), ("广场", "square"), ("胡同", "hutong"), ("老街", "old street"),
  ("古镇", "ancient town"), ("园林", "garden"), ("峡谷", "canyon"), ("瀑布", "waterfall"),
]


def _to_en_query(query: str) -> str:
  """把"城市+景点名"的中文查询转换为英文等价查询"""
  city_en = ""
  for cn, en in CITY_MAP.items():
    if cn in query:
      city_en = en
      break
  words = []
  for cn, en in LANDMARK_MAP:
    if cn in query and en not in words:
      words.append(en)
  return f"{city_en or 'China'} {' '.join(words[:2])}".strip()


def _search(query: str, access_key: str, timeout: float) -> Optional[str]:
  """调用Unsplash搜索API，命中返回URL，未命中/失败返回None"""
  try:
    resp = httpx.get(
      "https://api.unsplash.com/search/photos",
      params={"query": query, "per_page": 1, "orientation": "landscape"},
      headers={"Authorization": f"Client-ID {access_key}"},
      timeout=timeout,
    )
  except httpx.HTTPError as e:
    logger.debug("Unsplash 请求失败(%s): %s", query, str(e)[:80])
    return None
  if resp.status_code == 200:
    results = resp.json().get("results", [])
    if results:
      return results[0].get("urls", {}).get("regular")
    logger.debug("Unsplash 无结果(HTTP 200): %s", query)
  elif resp.status_code in (401, 403):
    logger.warning("Unsplash 鉴权/限流失败(HTTP %s)，跳过配图", resp.status_code)
  else:
    logger.debug("Unsplash 非预期状态(HTTP %s): %s", resp.status_code, query)
  return None


def fetch_attraction_image(query: str, timeout: float = 3.0) -> Optional[str]:
  """按关键词检索一张横向照片；先中文查询，空结果转英文重试；任何失败返回None"""
  key = (query or "").strip()
  if not key:
    return None
  with _cache_lock:
    if key in _cache:
      return _cache[key]

  access_key = get_settings().unsplash_access_key
  if not access_key:
    logger.debug("未配置UNSPLASH_ACCESS_KEY，跳过景点配图")
    with _cache_lock:
      _cache[key] = None
    return None

  url = None
  try:
    url = _search(key, access_key, timeout)
    if not url:
      en = _to_en_query(key)
      if en != key:
        url = _search(en, access_key, timeout)
  except Exception as e:
    logger.debug("Unsplash 检索异常: %s", str(e)[:80])

  with _cache_lock:
    _cache[key] = url
  return url


def fetch_attraction_images(queries: List[str], timeout: float = 3.0,
                            max_workers: int = 4) -> Dict[str, Optional[str]]:
  """批量并发检索配图，返回 {query: url_or_None}。
  用于一次性为所有景点并发取图，避免串行等待每个 5s 超时。"""
  queries = [q for q in (queries or []) if q]
  if not queries:
    return {}
  results: Dict[str, Optional[str]] = {}
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    future_map = {pool.submit(fetch_attraction_image, q, timeout): q for q in queries}
    for future in as_completed(future_map):
      q = future_map[future]
      try:
        results[q] = future.result()
      except Exception:
        results[q] = None
  return results
