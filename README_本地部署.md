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
      TA_APP_SECRET_KEY: ${TA_APP_SECRET_KEY:?请先设置 TA_APP_SECRET_KEY}
      TA_API_KEY: ${TA_API_KEY}
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
3. **API Key**: `${TA_API_KEY}` (通过 `http://localhost:8317/v1/models` 验证，用自己的密钥)
4. **推荐模型**:
   - `gemini-2.5-flash` (速度快、响应稳定)
   - `gemini-3.5-flash`
   - `claude-sonnet-4-6`

---

## 模型列表抓取白名单（`TA_MODELS_FETCH_ALLOWLIST`）

Web 界面「设置」里点"获取模型列表"时，后端会从 Base URL 拉取可用模型。出于 SSRF 防护，后端只允许抓取显式白名单里的主机，且**默认 fail-closed**（未配置一律返回「无法获取模型列表」）。

按部署形态配置：

- **Docker（容器内访问宿主机代理）**：设置 `TA_MODELS_FETCH_ALLOWLIST=host.docker.internal:8317`
- **裸机（本机直接跑，代理在本机）**：设置 `TA_MODELS_FETCH_ALLOWLIST=127.0.0.1:8317`
- **远程/公网代理**：设置 `TA_MODELS_FETCH_ALLOWLIST=<远程域名>:<端口>`

注意事项：

1. **必须写端口**。受信本地主机（`host.docker.internal` / `localhost` / `127.0.0.1` / `::1`）不写端口视为无效，防止"任意端口可探"。
2. **云元数据地址永远拦截**（`169.254.169.254` 等），任何白名单配置都不放行。
3. 白名单支持多个条目，用英文逗号分隔，例如 `TA_MODELS_FETCH_ALLOWLIST=host.docker.internal:8317,api.openai.com:443`。
4. 该接口的匿名访问只允许来自本机回环（`127.0.0.1`）。**Docker 场景通过映射端口从浏览器访问时，需要先登录（或使用 API Token）**，否则返回 401；裸机本机访问不受影响。

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
