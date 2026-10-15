"""高德地图 MCP Server（本地修复版）

基于 PyPI 包 amap-mcp-server 0.1.11 fork 而来。
修复点：删除 maps_direction_transit_integrated_by_coordinates 里的 `print(data)`
  —— 该行把高德响应 dict 的 Python repr（单引号）输出到 stdout，
     在 MCP stdio transport 下污染 JSON-RPC 协议通道，导致客户端解析炸掉。

用法：python amap_mcp_server_local/server.py stdio
依赖：fastmcp, requests, mcp （与 uvx amap-mcp-server 相同）
"""
import os
import sys
import argparse
from typing import Any, Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP


def get_api_key() -> str:
    api_key = os.getenv("AMAP_MAPS_API_KEY")
    if not api_key:
        raise ValueError("AMAP_MAPS_API_KEY environment variable is required")
    return api_key


AMAP_MAPS_API_KEY = get_api_key()

mcp = FastMCP("amap-maps")


@mcp.tool()
def maps_regeocode(location: str) -> Dict[str, Any]:
    """将一个高德经纬度坐标转换为行政区划地址信息"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/geocode/regeo",
            params={"key": AMAP_MAPS_API_KEY, "location": location},
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"RGeocoding failed: {data.get('info') or data.get('infocode')}"}
        return {
            "province": data["regeocode"]["addressComponent"]["province"],
            "city": data["regeocode"]["addressComponent"]["city"],
            "district": data["regeocode"]["addressComponent"]["district"],
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_geo(address: str, city: Optional[str] = None) -> Dict[str, Any]:
    """将详细的结构化地址转换为经纬度坐标"""
    try:
        params: Dict[str, Any] = {"key": AMAP_MAPS_API_KEY, "address": address}
        if city:
            params["city"] = city
        response = requests.get("https://restapi.amap.com/v3/geocode/geo", params=params)
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"Geocoding failed: {data.get('info') or data.get('infocode')}"}
        results = []
        for geo in data.get("geocodes", []):
            results.append({
                "country": geo.get("country"), "province": geo.get("province"),
                "city": geo.get("city"), "citycode": geo.get("citycode"),
                "district": geo.get("district"), "street": geo.get("street"),
                "number": geo.get("number"), "adcode": geo.get("adcode"),
                "location": geo.get("location"), "level": geo.get("level"),
            })
        return {"return": results}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_ip_location(ip: str) -> Dict[str, Any]:
    """IP 定位"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/ip",
            params={"key": AMAP_MAPS_API_KEY, "ip": ip},
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"IP Location failed: {data.get('info') or data.get('infocode')}"}
        return {
            "province": data.get("province"), "city": data.get("city"),
            "adcode": data.get("adcode"), "rectangle": data.get("rectangle"),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_weather(city: str) -> Dict[str, Any]:
    """根据城市名称或者标准adcode查询指定城市的天气"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={"key": AMAP_MAPS_API_KEY, "city": city, "extensions": "all"},
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"Get weather failed: {data.get('info') or data.get('infocode')}"}
        forecasts = data.get("forecasts", [])
        if not forecasts:
            return {"error": "No forecast data available"}
        return {"city": forecasts[0]["city"], "forecasts": forecasts[0]["casts"]}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_text_search(keywords: str, city: str = "", citylimit: str = "false") -> Dict[str, Any]:
    """关键词搜索 API，返回 POI 列表"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={"key": AMAP_MAPS_API_KEY, "keywords": keywords,
                    "city": city, "citylimit": citylimit},
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"Text Search failed: {data.get('info') or data.get('infocode')}"}
        pois = []
        for poi in data.get("pois", []):
            pois.append({
                "id": poi.get("id"), "name": poi.get("name"),
                "address": poi.get("address"), "typecode": poi.get("typecode"),
            })
        suggestion_cities = []
        for c in data.get("suggestion", {}).get("cities", []) or []:
            suggestion_cities.append({"name": c.get("name")})
        return {
            "suggestion": {"keywords": data.get("suggestion", {}).get("keywords", []),
                           "cities": suggestion_cities},
            "pois": pois,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_search_detail(id: str) -> Dict[str, Any]:
    """查询关键词搜或者周边搜获取到的 POI ID 的详细信息"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/detail",
            params={"key": AMAP_MAPS_API_KEY, "id": id},
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"Get poi detail failed: {data.get('info') or data.get('infocode')}"}
        if not data.get("pois"):
            return {"error": "No POI found"}
        poi = data["pois"][0]
        result = {
            "id": poi.get("id"), "name": poi.get("name"),
            "location": poi.get("location"), "address": poi.get("address"),
            "business_area": poi.get("business_area"), "city": poi.get("cityname"),
            "type": poi.get("type"), "alias": poi.get("alias"),
        }
        if poi.get("biz_ext"):
            result.update(poi["biz_ext"])
        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_direction_driving_by_coordinates(origin: str, destination: str) -> Dict[str, Any]:
    """驾车路径规划（按坐标）"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/driving",
            params={"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination},
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] != "1":
            return {"error": f"Direction Driving failed: {data.get('info') or data.get('infocode')}"}
        paths = []
        for path in data["route"]["paths"]:
            steps = []
            for step in path["steps"]:
                steps.append({
                    "instruction": step.get("instruction"), "road": step.get("road"),
                    "distance": step.get("distance"), "orientation": step.get("orientation"),
                    "duration": step.get("duration"),
                })
            paths.append({
                "path": path.get("path"), "distance": path.get("distance"),
                "duration": path.get("duration"), "steps": steps,
            })
        return {"route": {
            "origin": data["route"]["origin"],
            "destination": data["route"]["destination"],
            "paths": paths,
        }}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


@mcp.tool()
def maps_direction_transit_integrated_by_coordinates(
    origin: str, destination: str, city: str, cityd: str
) -> Dict[str, Any]:
    """综合公交（火车、公交、地铁）路径规划（按坐标）。

    本地修复版：删除了原包第563行的 `print(data)` ——
    该行把高德响应 dict repr 输出到 stdout，在 MCP stdio 下污染协议通道。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/transit/integrated",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin, "destination": destination,
                "city": city, "cityd": cityd,
            },
        )
        response.raise_for_status()
        data = response.json()
        # ↓ 原来这里有 print(data)，已删除（本文件就是修复版）
        if data.get("status") != "1":
            return {"error": f"Direction Transit Integrated failed: {data.get('info') or data.get('infocode')}"}

        route_data = data.get("route")
        if not isinstance(route_data, dict):
            return {"error": "No route data available"}

        transits: List[Dict[str, Any]] = []
        transits_data = route_data.get("transits")
        if isinstance(transits_data, list):
            for transit in transits_data:
                if not isinstance(transit, dict):
                    continue
                segments: List[Dict[str, Any]] = []
                for segment in (transit.get("segments") or []):
                    if not isinstance(segment, dict):
                        continue
                    walking = segment.get("walking") if isinstance(segment.get("walking"), dict) else {}
                    walking_steps = []
                    for step in (walking.get("steps") or []):
                        if isinstance(step, dict):
                            walking_steps.append({
                                "instruction": step.get("instruction"),
                                "road": step.get("road"),
                                "distance": step.get("distance"),
                                "action": step.get("action"),
                                "assistant_action": step.get("assistant_action"),
                                "polyline": step.get("polyline"),
                            })
                    bus = segment.get("bus") if isinstance(segment.get("bus"), dict) else {}
                    buslines = []
                    for bl in (bus.get("buslines") or []):
                        if not isinstance(bl, dict):
                            continue
                        dep_stop = bl.get("departure_stop") if isinstance(bl.get("departure_stop"), dict) else {}
                        arr_stop = bl.get("arrival_stop") if isinstance(bl.get("arrival_stop"), dict) else {}
                        via_stops = []
                        for s in (bl.get("via_stops") or []):
                            if isinstance(s, dict):
                                via_stops.append({"name": s.get("name")})
                        buslines.append({
                            "name": bl.get("name"),
                            "departure_stop": {"name": dep_stop.get("name")},
                            "arrival_stop": {"name": arr_stop.get("name")},
                            "distance": bl.get("distance"),
                            "duration": bl.get("duration"),
                            "via_stops": via_stops,
                            # 高德 polyline 路径（原包里漏了，这里补齐）
                            "polyline": bl.get("polyline"),
                        })
                    entrance = segment.get("entrance") if isinstance(segment.get("entrance"), dict) else {}
                    exit_data = segment.get("exit") if isinstance(segment.get("exit"), dict) else {}
                    railway = segment.get("railway") if isinstance(segment.get("railway"), dict) else {}

                    segments.append({
                        "walking": {
                            "origin": walking.get("origin"),
                            "destination": walking.get("destination"),
                            "distance": walking.get("distance"),
                            "duration": walking.get("duration"),
                            "steps": walking_steps,
                            "polyline": walking.get("polyline"),  # 补齐 walking polyline
                        },
                        "bus": {"buslines": buslines},
                        "entrance": {"name": entrance.get("name")},
                        "exit": {"name": exit_data.get("name")},
                        "railway": {
                            "name": railway.get("name"),
                            "trip": railway.get("trip"),
                        },
                    })

                transits.append({
                    "duration": transit.get("duration"),
                    "distance": transit.get("distance"),
                    "walking_distance": transit.get("walking_distance"),
                    "segments": segments,
                })

        return {"route": {
            "origin": route_data.get("origin"),
            "destination": route_data.get("destination"),
            "distance": route_data.get("distance"),
            "transits": transits,
        }}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amap MCP Server (local patched)")
    parser.add_argument('transport', nargs='?', default='stdio',
                        choices=['stdio', 'sse', 'streamable-http'])
    args = parser.parse_args()
    mcp.run(transport=args.transport)
