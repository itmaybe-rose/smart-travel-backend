# Hugging Face Spaces (Docker) 部署镜像
# 基础镜像：Python 3.11（stable，FastAPI/uvicorn/LangGraph 全兼容）
FROM python:3.11-slim

# 工作目录（HF Spaces 会自动将 /spaces/<your-space-name> 挂载为工作目录）
WORKDIR /app

# 系统依赖：requests 用 SSL，httpx/aiohttp 可能需要编译
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，利用 Docker 缓存层（依赖不变时不重装）
COPY requirements.txt .

# 安装 Python 依赖（--no-cache-dir 减镜像体积）
RUN pip install --no-cache-dir -r requirements.txt

# 复制全部后端源码（.dockerignore 已排除 venv、.env、__pycache__）
COPY app/ ./app/
COPY amap_mcp_server_local/ ./amap_mcp_server_local/

# 环境变量：HF Spaces 要求监听 7860 端口
ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 健康检查（HF Spaces 会定期访问）
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 启动 FastAPI（run.py 默认 reload=True，生产环境必须用 uvicorn 直启）
# 注意：HF Spaces 免费层是单 CPU，不要开多 worker
EXPOSE 7860
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
