"""FastAPI主应用"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings, print_config, validate_config
from ..agents.factory import get_trip_planner_agent
from .routes import trip

settings = get_settings()

# 全局日志配置：控制台输出，INFO 级别，带模块名与时间戳
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
  """应用生命周期：启动时预热规划器实例，把MCP连接/图编译的开销从首请求移走。"""
  print("\n" + "=" * 60)
  print(f"🚀 {settings.app_name} v{settings.app_version}")
  print("=" * 60)

  print_config()
  try:
    validate_config()
    print("✅ 配置验证通过")
  except Exception as e:
    print(f"\n❌ 配置验证失败: {e}")
    raise

  # 预热规划器：懒加载单例首次构建较重（MCP握手 + 工具发现 + 图编译），
  # 在启动期完成后，首请求直接进业务流程，体验更稳。
  print("⏳ 预热旅行规划器（MCP连接 + 工具发现 + 图编译）...")
  try:
    get_trip_planner_agent()
    print("✅ 旅行规划器预热完成")
  except Exception as e:
    print(f"⚠️  规划器预热失败（首次请求时会再尝试）: {e}")

  print("\n" + "=" * 60)
  print("📚 API文档: http://localhost:8000/docs")
  print("=" * 60 + "\n")
  yield


# 创建FastAPI应用
app = FastAPI(
  title=settings.app_name,
  version=settings.app_version,
  description=settings.app_description,
  docs_url="/docs",
  redoc_url="/redoc",
  lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=settings.get_cors_origins_list(),
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# 注册路由
app.include_router(trip.router, prefix="/api")


@app.get("/")
async def root():
  """根路径"""
  return {
    "name": settings.app_name,
    "version": settings.app_version,
    "status": "running",
    "docs": "/docs",
  }


@app.get("/health")
async def health():
  """健康检查"""
  return {
    "status": "healthy",
    "service": settings.app_name,
    "version": settings.app_version,
  }
