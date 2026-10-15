'''数据模型定义'''

from typing import List,Optional,Union
from pydantic import BaseModel,Field,field_validator,model_validator


'''
Field是给字段加约束和文档信息的包装函数
...表示必填字段
description="字段描述"
example="示例值"
'''
#===========请求模型==========

class TripRequest(BaseModel):
  '''旅行规划请求'''
  city:str = Field(...,description="目的地城市",example="北京")
  start_date:str = Field(...,description="开始日期 YYYY-MM-DD",example="2025-06-01")
  end_date:str = Field(...,description="结束日期 YYYY-MM-DD",example="2025-06-03")
  travel_days:int = Field(...,description="旅行天数",ge=1,le=30,example="2025-06-03")
  #ge代表大于等于，le代表小于等于
  transportation:str = Field(...,description="交通方式",example="公共交通")
  accommodation:str = Field(...,description="住宿偏好",example="经济型酒店")
  preferences:List[str] = Field(default=[],description="旅行偏好标签",example=["历史文化","美食"])
  free_text_input:Optional[str] = Field(default="",description="额外要求",example="希望多安排一点博物馆")

#=========响应模型=============
class Location(BaseModel):
  '''地理位置'''
  longitude:float = Field(...,description="经度")
  latitude:float = Field(...,description="纬度")

class Attraction(BaseModel):
  '''景点信息'''
  name:str = Field(...,description="景点名称")
  address:str = Field(...,description="地址")
  location:Location = Field(...,description="经纬度坐标")
  visit_duration:int = Field(...,description="建议游览时间（分钟）")
  description:str = Field(...,description="景点描述")
  category:Optional[str] = Field(default="景点",description="景点类别")
  ticket_price:int = Field(default=0,description="门票价格（元）")
  image_url:Optional[str] = Field(default=None,description="景点配图URL（Unsplash检索，可能为空）")

class Meal(BaseModel):
  '''餐饮信息'''
  type:str = Field(...,description="餐饮类型:breakfast/lunch/dinner")
  name:str = Field(...,description="餐饮名称")
  description:Optional[str] = Field(default=None,description="餐饮描述")
  estimated_cost:int = Field(default=0,description="预估费用（元）")

class Hotel(BaseModel):
  '''酒店信息'''
  name:str = Field(...,description="酒店名称")
  address:str = Field(default="",description="酒店地址")
  location:Optional[Location] = Field(default=None,description="酒店位置") 
  price_range:str = Field(default="",description="酒店价格范围")
  estimated_cost:int = Field(default=0,description="预估费用(元/晚)")

class TransportSegment(BaseModel):
  '''景点间交通段（高德路线规划真实数据）'''
  from_name:str = Field(...,description="起点景点名称")
  to_name:str = Field(...,description="终点景点名称")
  distance_km:float = Field(default=0,description="距离（公里）")
  duration_min:int = Field(default=0,description="耗时（分钟）")
  mode:str = Field(default="",description="交通方式描述（驾车/公交/步行）")
  cost:int = Field(default=0,description="预估费用（元）")
  polyline: str = Field(default="", description="高德真实路线折线（lon,lat;lon,lat...）")

class DayPlan(BaseModel):
  '''单日行程'''
  date:str = Field(...,description="日期 YYYY-MM-DD")
  day_index:int = Field(...,description="第几天(从0开始)")
  description:str = Field(...,description="当日行程描述")
  transportation:str = Field(default="",description="交通方式")
  accommodation:str = Field(default="",description="住宿")
  hotel:Optional[Hotel] = Field(default=None,description="推荐酒店")
  attractions:List[Attraction] = Field(default=[],description="景点列表")
  meals:List[Meal] = Field(default=[],description="餐饮列表")
  transport_segments:List[TransportSegment] = Field(default=[],description="景点间交通段（高德路线规划）")

class WeatherInfo(BaseModel):
  '''天气信息'''
  date:str = Field(...,description="日期 YYYY-MM-DD")
  day_weather:str = Field(default="",description="白天天气")
  night_weather:str = Field(default="",description="晚上天气")
  day_temp:Union[int,str] = Field(default=0,description="白天温度")
  night_temp:Union[int,str] = Field(default=0,description="晚上温度")
  #Union[int,str] 允许输入整数或字符串表示的温度值

  #注册验证器(处理特殊格式)
  @field_validator("day_temp","night_temp",mode="before") # 验证温度字段，将字符串转换为整数
  @classmethod # Pydantic验证器必须是类方法
  def parse_temperature(cls,v):
    '''解析温度，移除单位（如℃）'''
    if isinstance(v,str):
      v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
      try:
        return int(v)
      except ValueError:
        return 0
    return v
  
class Budget(BaseModel):
  '''预算信息'''
  total_attractions:int = Field(default=0,description="景点门票总费用")
  total_hotels:int = Field(default=0,description="酒店总费用")
  total_meals:int = Field(default=0,description="餐饮总费用")
  total_transportation:int = Field(default=0,description="交通总费用")
  total:int = Field(default=0,description="总费用")

class TripPlan(BaseModel):
  '''旅行计划'''
  city:str = Field(...,description="目的地城市")
  start_date:str = Field(...,description="开始日期 YYYY-MM-DD")
  end_date:str = Field(...,description="结束日期 YYYY-MM-DD")
  days:List[DayPlan] = Field(...,description="每日行程")
  weather_info:List[WeatherInfo] = Field(default=[],description="天气信息")
  overall_suggestion:str = Field(...,description="总体建议")
  budget:Optional[Budget] = Field(default=None,description="预算信息")
  route_map_url: Optional[str] = Field(default=None, description="全程路线总览图相对URL（后端代理高德staticmap）")

  @model_validator(mode="before") # LLM偶尔会按提示词旧示例输出复数overall_suggestions，这里统一映射
  @classmethod
  def _alias_overall_suggestion(cls,data):
    if isinstance(data,dict) and "overall_suggestions" in data:
      data.setdefault("overall_suggestion",data.pop("overall_suggestions"))
    return data

class TripPlanResponse(BaseModel):
  '''旅行计划响应'''
  success:bool = Field(...,description="是否成功")
  message:str  = Field(default="",description="消息") 
  #default="" 表示默认值为空字符串，用于表示成功或失败的消息
  data:Optional[TripPlan] = Field(default=None,description="旅行计划数据")
  #Optional[TripPlan] 允许返回None值，表示失败或无数据
