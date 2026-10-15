"""应用启动脚本"""
import uvicorn
from app.config import get_settings

if __name__ == "__main__":
  settings = get_settings()
  uvicorn.run(
    "app.api.main:app",
    host=settings.host,
    port=settings.port,
    reload=True # 开启热重载,代码修改后自动重启服务器
  )
