# TradingAgents-AShare 本地部署与使用指南

本文档记录了在 macOS (Apple Silicon arm64) 上通过 Docker 部署 **TradingAgents-AShare** 并接入本地 **CLIProxyAPI** 的完整操作与运维说明。

---

## 目录
1. [系统与架构概览](#系统与架构概览)
2. [环境配置与服务启动](#环境配置与服务启动)
3. [LLM 接入配置说明](#llm-接入配置说明)
4. [常用运维命令（启动/停止/重启/日志）](#常用运维命令)
5. [实测验证数据（贵州茅台 600519）](#实测验证数据)
6. [常见报错及处理方法](#常见报错及处理方法)
7. [需手动完成的事项](#需手动完成的事项)

---

## 系统与架构概览

- **项目仓库**: `https://github.com/KylinMountain/TradingAgents-AShare`
- **项目目录**: `~/Documents/TradingAgents-AShare`
- **部署方式**: Docker 容器化单体部署（含 FastAPI 后端、React 静态前端与后台 Task 调度）
- **Web UI 访问地址**: `http://localhost:8001`
- **底层数据库**: SQLite (`/app/data/tradingagents.db` -> 宿主机 `./data/tradingagents.db`)
- **LLM 网关**: 宿主机 CLIProxyAPI (`http://localhost:8317`)
- **容器访问宿主机地址**: `http://host.docker.internal:8317/v1`

---

## 环境配置与服务启动

### 配置文件位置
- **Docker Compose 配置**: `~/Documents/TradingAgents-AShare/docker-compose.yml`
- **持久化数据目录**: `~/Documents/TradingAgents-AShare/data/`

### `docker-compose.yml` 关键环境变量配置
```yaml
services:
  app:
    image: tradingagents-ashare:latest
    container_name: tradingagents-ashare
    restart: always
    ports:
      - "8001:8000"
    volumes:
      - ./data:/app/data
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      DATABASE_URL: sqlite:///./data/tradingagents.db
      TA_APP_SECRET_KEY: g48dSK9v1pXZmQ3L8eR2tB7yN0uW5c6A8x9Z1v2W3e4R5t6Y
      TA_API_KEY: sk-Kmd9ysjg9plNpXE98
      TA_BASE_URL: http://host.docker.internal:8317/v1
      TA_LLM_PROVIDER: openai
      TA_LLM_QUICK: gemini-2.5-flash
      TA_LLM_DEEP: gemini-2.5-flash
```

---

## 常用运维命令

### 启动服务
```bash
cd ~/Documents/TradingAgents-AShare
docker compose up -d
```

### 停止服务
```bash
cd ~/Documents/TradingAgents-AShare
docker compose down
```

### 查看实时运行日志
```bash
docker logs -f tradingagents-ashare
```

### 查看容器状态
```bash
docker ps -a | grep tradingagents-ashare
```

---

## LLM 接入配置说明

1. **接口兼容模式**: OpenAI 兼容 (Custom Endpoint / `openai`)
2. **容器内 Base URL**: `http://host.docker.internal:8317/v1`
3. **API Key**: `sk-Kmd9ysjg9plNpXE98` (通过 `http://localhost:8317/v1/models` 验证)
4. **推荐模型**:
   - `gemini-2.5-flash` (速度快、响应稳定)
   - `gemini-3.5-flash`
   - `claude-sonnet-4-6`

---

## 实测验证数据（贵州茅台 600519）

在 Web 界面对 **600519 (贵州茅台)** 进行了真实完整分析跑通，结果如下：

- **分析股票**: 贵州茅台 (`600519.SH`)
- **研究深度**: 快捷分析 (Lowest Depth)
- **数据源**: AKShare (免费源)
- **总耗时**: 10 分钟 27 秒 (627 秒)
- **分析流程验证**:
  1. **基本面分析师 (Fundamentals Analyst)**: 输出财报周期盈利与收入质量评估。
  2. **技术面分析师 (Market Analyst)**: 输出 14 天 K 线、EMA、RSI、MACD 动能分析。
  3. **多空辩论 (Bull vs Bear Debate)**: Bull Analyst (短线偏多) 与 Bear Analyst (中线偏空) 针对基本面下行与技术回调展开激烈辩论。
  4. **投研经理 (Research Manager)**: 综合评估辩论证据，认定当前反弹为诱多陷阱，裁定看空。
  5. **交易员与风控 (Trader & Risk Manager)**: 拟定卖出方案，设置止损价 `1355.0` 元，第一目标减仓价 `1275.5` 元，风控限制初始仓位 <= 25%。
- **最终决策结论**: 
  - **方向**: 卖出 (看空 / Bearish)
  - **置信度**: 85%
  - **目标价**: 1275.50 元
  - **止损价**: 1355.00 元

---

## 常见报错及处理方法

### 1. 容器内无法连接 LLM API (`Connection refused` 或 `Cannot reach host`)
- **原因**: 容器内访问 `localhost` 会指向容器自身而非宿主机。
- **解决办法**: 确保 Base URL 填的是 `http://host.docker.internal:8317/v1`，且 `docker-compose.yml` 中包含 `extra_hosts: - "host.docker.internal:host-gateway"`。

### 2. HTTP 401 Unauthorized 或 Missing API Key
- **原因**: CLIProxyAPI 需要在请求头包含 Bearer Token 鉴权。
- **解决办法**: 在 `.env` 或 `docker-compose.yml` 中配置正确的 `TA_API_KEY`（可在 `http://localhost:8317/management.html` 查验）。

### 3. Web 端口冲突 (`Port 8000 already in use`)
- **原因**: 本地已运行 TradingAgents-CN 后端占用了 8000 端口。
- **解决办法**: TradingAgents-AShare 映射为宿主机端口 `8001:8000`，请在浏览器访问 `http://localhost:8001`。

### 4. AKShare 接口限流或网络超时
- **原因**: 频繁调用东方财富/新浪财经接口触发了免费数据源频控。
- **解决办法**: 系统内置重试与降级机制，稍等几分钟后重新提交任务即可。

---

## 仍需您手动完成的事项

1. **浏览器访问**: 启动后在浏览器打开 `http://localhost:8001` 体验完整 Web 界面。
2. **模型选配调整 (可选)**: 若需在 UI 界面手动修改默认模型或切换为 `claude-sonnet-4-6` / `gemini-3.5-flash`，可在 Web 界面的设置页面中进行微调。
3. **管理 CLIProxyAPI 密钥**: 若您在 `http://localhost:8317/management.html` 中重置或更换了密钥，请同步更新 `docker-compose.yml` 中的 `TA_API_KEY` 并重启容器。
