FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS runtime
WORKDIR /app

# 安装基础系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 利用 uv 同步依赖
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 拷贝后端源码与脚本
COPY api/ ./api/
COPY tradingagents/ ./tradingagents/
COPY scheduler/ ./scheduler/
COPY docker-entrypoint.py ./docker-entrypoint.py

# 安装项目本身
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 从前端构建阶段拷贝产物（clean checkout 无需预生成 frontend/dist）
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# 暴露端口
EXPOSE 8000

ARG VERSION=dev
ENV APP_VERSION=${VERSION}

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["/app/docker-entrypoint.py"]
