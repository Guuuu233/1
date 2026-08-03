# TA_MODELS_FETCH_ALLOWLIST 配置说明

> 适用成员 / 负责人：所有部署与运维人员
> 相关代码：`api/main.py`（`_MODELS_FETCH_ALLOWLIST_ENV` 及 `_fetch_available_models` 系列）
> 关联接口：`POST /v1/models/fetch`（模型列表抓取，SSRF 加固）

## 概述

`TA_MODELS_FETCH_ALLOWLIST` 是 `POST /v1/models/fetch` 接口的 **SSRF 防护白名单**。
该接口允许用户在「模型设置」中填写自定义 Base URL 并抓取可用模型列表；
为避免服务端被诱导访问任意内网/云元数据地址（SSRF），系统要求：

1. 目标 Base URL 的 **host 必须命中该白名单**；
2. 目标 host 解析出的 **所有 IP 必须是公网/全局 IP**（私网、回环、链路本地、云元数据等一律拒绝）；
3. 请求失败时**只返回通用错误**，不泄露内部细节（fail-closed）。

**未配置该变量时，模型列表抓取功能整体禁用**（返回 `ok: false`），属于预期行为。

## 配置方式

环境变量名：`TA_MODELS_FETCH_ALLOWLIST`

- Docker Compose：在 `docker-compose.yml` 的 `app.environment` 中增加
  `TA_MODELS_FETCH_ALLOWLIST: ${TA_MODELS_FETCH_ALLOWLIST}`，并在 `.env` 文件中填写取值。
- 源码部署：在启动后端进程的环境中导出该变量即可。

示例：

```bash
# 允许 models.example.com 的任意端口 + gateway.example.com 仅 8443 端口
TA_MODELS_FETCH_ALLOWLIST="models.example.com;gateway.example.com:8443"
```

## 值格式

取值是一个**条目列表**，条目之间用英文逗号 `,` 或分号 `;` 分隔，可混合使用。
每个条目支持三种形式：

| 形式 | 含义 | 示例 |
|------|------|------|
| `host` | 允许该主机的**任意端口** | `models.example.com` |
| `host:port` | 仅允许该主机的**指定端口**（1–65535） | `gateway.example.com:8443` |
| `[ipv6]` / `[ipv6]:port` | IPv6 主机，可选端口 | `[2001:db8::1]:8080` |

解析规则：

- 条目中的空格会被忽略（逐条 `strip`）；空条目自动跳过。
- host 匹配**不区分大小写**，并忽略末尾的点号（`.`）。
- 条目中不允许出现 `/ ? # @` 等字符（视为非法白名单配置，直接拒绝请求）。
- 端口必须是纯数字且落在 1–65535；IPv6 须用 `[...]` 包裹。

### 端口匹配语义

- 白名单条目**不带端口**（如 `models.example.com`）：该 host 的任意端口（含默认 80/443）均被允许。
- 白名单条目**带端口**（如 `gateway.example.com:8443`）：仅该精确端口被允许，请求使用其他端口会被拒绝。

## 请求校验流程

接口收到 `base_url` 后按以下顺序确定抓取目标：

1. 请求体 `base_url`（前端传入）；
2. `provider_id` 对应的 Provider 配置 `base_url`；
3. 当前用户的 LLM 配置 `custom_base_url`；
4. 兜底默认值 `http://host.docker.internal:8317/v1`。

随后执行的安全校验（全部通过才发起请求）：

| 步骤 | 规则 | 失败结果 |
|------|------|----------|
| 白名单配置存在 | `TA_MODELS_FETCH_ALLOWLIST` 未设置/为空 → 拒绝 | `models fetch allowlist is not configured` |
| URL 合法性 | scheme 仅允许 `http`/`https`；禁止 user:pass、query、fragment、控制字符；host 非空；端口 1–65535 | 拒绝 |
| host 白名单 | 请求 host 必须命中白名单，且端口匹配 | `host is not allowlisted` |
| DNS 解析与 IP 过滤 | 所有解析结果必须为公网/全局 IP，否则拒绝 | `resolved address is blocked` |
| 连接建立 | 用解析到的安全 IP 建连；HTTPS 保留原 hostname 作为 SNI | — |

解析后的连接会**固定到解析出的安全 IP**，避免 DNS rebinding 类绕过；HTTPS 仍以原始 hostname 发送 SNI。

### 被拦截的 IP 类别

以下地址解析结果一律拒绝（fail-closed，任一解析地址命中即拒绝）：

- 云元数据地址：`169.254.169.254`、`fd00:ec2::254`
- 私网、回环（127.0.0.0/8、`::1`）、链路本地（169.254.0.0/16、`fe80::/10`）
- 保留、未指定、组播、非全局（non-global）地址
- IPv4-mapped IPv6 地址会先还原为 IPv4 再按上述规则判断

## 超时与响应语义

- 抓取超时：固定 **8 秒**（`_MODELS_FETCH_TIMEOUT_SECONDS`）。
- 成功：`GET <base_url>/v1/models`，返回
  `{"ok": true, "models": [...], "count": N, "url": "<实际请求的 /v1/models 地址>"}`；
  `models` 已去重并排序，并会自动同步角色路由的模型 profile。
- 失败（白名单未命中 / IP 被拦 / DNS 失败 / 超时 / 非 2xx / 解析异常等）：
  **统一返回 HTTP 200 +** `{"ok": false, "error": "无法获取模型列表", "models": [], "count": 0}`，
  具体原因仅写入服务端日志，不向前端暴露。

## 安全说明

- **缺省即禁用**：未配置该变量时，`/v1/models/fetch` 不会发起任何外部请求（fail-closed），
  不会因为「没配置白名单」而放开访问。
- **通用错误**：所有失败场景对外只返回「无法获取模型列表」，避免向调用方泄露内网拓扑/解析细节。
- **白名单 ≠ 内网豁免**：即使 host 被加入白名单，解析出的私网/云元数据地址仍会被拦截。
- 接口需要携带 API Token（`Authorization: Bearer <TOKEN>`）访问。

## 部署注意（假设 / 待确认）

- 默认兜底地址 `http://host.docker.internal:8317/v1` 是本地模型网关的常见位置；
  但该 host 在 Docker 环境下通常解析为**私网地址**，会被「IP 过滤」规则拦截。
  若你的模型网关部署在宿主机/内网，需要：

  1. 将网关地址加入白名单（如 `TA_MODELS_FETCH_ALLOWLIST=host.docker.internal:8317`），**并且**
  2. 确保其解析地址能被公网访问，或后续调整 IP 过滤策略（本轮不在本文档范围内，待确认）。

- 建议**仅白名单真正需要使用的模型网关 host**，最小化 SSRF 暴露面。
