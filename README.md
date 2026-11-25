# Qiye RAG · 企业知识库 Agentic RAG

面向企业制度、技术文档和产品资料的中文知识库问答项目。围绕“资料导入 → 授权检索 → 多轮推理 → 引用校验 → 效果评测”构建完整流程，提供 Web 上传、异步索引和带来源的多轮问答。

**核心技术：Python · LangGraph · LangChain · Milvus · BM25 · FastAPI · Celery · Redis · PostgreSQL**

## 项目亮点

- **中文混合检索**：jieba 分词与 BM25Plus 负责关键词匹配，Milvus 向量检索补充语义召回，RRF 融合后统一 Rerank。复杂问题可拆为多个查询，融合候选后再筛选证据。
- **有边界的 Agent 工作流**：Router 理解问题与历史指代，Retriever 查找证据，Researcher 生成答案，Reviewer 检查引用和事实；证据不足时补充检索，默认最多两轮、三个子查询。
- **答案可追溯**：保留文档编号、页码或章节、Chunk 编号；确定性检查引用编号，再由模型审核事实和回答完整性。未通过审核的草稿不会作为最终答案返回。
- **授权先于召回**：按知识库、角色、资料权限和现行状态约束检索语料；普通用户查询 public/internal，HR 可访问 restricted。会话范围绑定账号与知识库，历史用于问题理解。
- **可恢复的增量索引**：内容与配置摘要识别变化；先写入新版本，再原子提交 JSON 清单激活，最后清理旧版本；pending 日志辅助异常恢复，避免未激活版本进入应用检索。
- **异步上传与交互反馈**：PDF、DOCX、Markdown 支持单文件和批量上传；Celery + Redis 执行索引，前端轮询任务状态，SSE 返回节点进度与审核后的答案。
- **可测量的优化**：评测代码覆盖文档/Chunk 召回率、MRR、权限泄漏、拒答及引用召回；逐阶段记录耗时、逻辑模型调用数和 API 返回的 Token usage。支持离线 BM25 基线与真实服务对照。


## 技术栈

| 层次 | 技术 | 用途 |
| --- | --- | --- |
| 文档处理 | pypdf、python-docx、PyYAML、LangChain Text Splitters | 解析 PDF/DOCX/Markdown，保留来源与结构 |
| 检索 | jieba、rank-bm25、Milvus、LangChain Milvus | 关键词与语义召回、元数据过滤 |
| 模型接入 | LangChain OpenAI、httpx | DeepSeek 兼容聊天接口、百炼 Embedding/Rerank |
| Agent | LangGraph、Pydantic | 条件路由、结构化输出、审核和重试边界 |
| 会话持久化 | LangGraph SQLite Checkpointer | 节点状态保存与中断恢复 |
| 后端与前端 | FastAPI、Uvicorn、原生 HTML/JavaScript | 登录、上传、任务轮询、SSE 问答，无前端编译步骤 |
| 异步任务 | Celery、Redis、filelock | 文档索引任务与同知识库读写协调 |
| 账号与记录 | SQLAlchemy 2、PostgreSQL、Argon2 | 用户、Token 摘要、上传记录与密码哈希 |
| 验证 | pytest、可选 Optuna | 回归测试、评测和参数搜索 |


## 快速开始

### 1. 获取项目并安装依赖

需要 Python 3.11+。以下以 Windows PowerShell 为例，所有命令在项目根目录执行：

```powershell
git clone https://github.com/wzw41068855-cpu/qiye_rag2.git
cd qiye_rag2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
```

每个新终端都应激活同一环境，并设置上面的 PYTHONPATH。根目录 run_web.py 与 run_worker.py 自带 src 路径处理；模块命令与测试仍需 PYTHONPATH，或在包含 pyproject.toml 时执行 `python -m pip install -e .`。

requirements.txt 包含运行依赖和 pytest，使用版本范围而非完整锁定；它不安装 Redis、PostgreSQL 或 Milvus 服务。


### 2. 准备外部服务

| 服务 | 本地常用地址 | 要求 |
| --- | --- | --- |
| Milvus | http://localhost:19530 | 可连接的向量数据库 |
| Redis | redis://localhost:6379/0 | Web 与 worker 使用同一个地址 |
| PostgreSQL | localhost:5432 | 预先创建数据库与可用账号，程序负责建表 |

可使用已有本地服务或自行通过 Docker 部署。仓库不提供一键部署编排。Milvus 使用固定 schema：主键、text、vector 和 metadata JSON；不兼容的旧集合需使用新集合重建，程序不会自动迁移。

### 3. 配置模型与连接

首次配置时复制模板；已有 .env 请直接编辑，避免覆盖：

```powershell
Copy-Item .env.example .env
```

填写项目根目录 .env，下面均为占位示例：

```dotenv
DASHSCOPE_API_KEY=替换为百炼密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
RERANK_API_KEY=替换为百炼密钥
RERANK_MODEL=qwen3-rerank
RERANK_API_BASE=https://dashscope.aliyuncs.com/api/v1

DEEPSEEK_API_KEY=替换为聊天模型密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
CHAT_MODEL=替换为账号可用的聊天模型名

MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
REDIS_URI=redis://localhost:6379/0
DATABASE_URI=postgresql+psycopg://rag_user:替换为密码@localhost:5432/qiye_rag
```


### 4. 初始化账号

数据库须已存在；密码通过终端隐藏输入，无默认账号、无公开 HR 注册：

```powershell
python -m qiye_rag.run_time.users hr_admin --role hr
python -m qiye_rag.run_time.users alice --role user
```

### 5. 启动 worker 与 Web

打开两个终端，使用同一项目、解释器、.env 和共享资料目录：

```powershell
# 终端一：Windows 使用 solo 单进程 worker
python run_worker.py
```

```powershell
# 终端二
python run_web.py
```

访问 http://127.0.0.1:8000/login；API 文档位于 http://127.0.0.1:8000/docs。

1. 使用 HR 账号登录，选择知识库，上传 PDF/DOCX/Markdown，设置 internal 或 restricted。
2. 等待索引任务 SUCCESS，再查看文档与 Chunk 预览；上传完成不代表索引完成。
3. 输入问题，例如“员工试用期多久？”；同一会话继续追问“转正需要哪些材料？”。
4. 查看节点进度、最终答案及来源引用；切换普通用户，验证受限资料不会进入检索范围。

Web 上传库独立管理，不自动导入 data 目录，也不接管 CLI 索引。批量选目录时 README.md 等支持格式也会作为资料上传，应只选择需要问答的文件。


## 测试与评测

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m pytest tests -q
```

默认回归使用模型和向量服务替身，ORM 使用 SQLite 测试环境，不消耗真实模型配额；真实服务冒烟需显式启用对应 RAG_*_SMOKE 开关。

评测代码定义 30 道题，覆盖事实查询、精确匹配、语义改写、多跳推理、版本冲突、权限隔离、无法回答与多轮指代。每题包含 gold_document_ids、gold_chunk_ids、参考事实、预期引用和是否可回答。

```powershell
#rag评估
python -m qiye_rag.evaluation --mode bm25

# 根据资料与代码中的事实锚点重建 Gold；生成后仍须人工复核
python -m qiye_rag.evaluation --build --mode bm25

# 真实 Embedding / Rerank / Milvus / 聊天模型，会产生 API 费用
python -m qiye_rag.evaluation --mode agent
```

结果写入 eval/runs 下的新目录，包括逐题结果、阶段事件、元数据和汇总。


## 目录结构

```text
.
├── run_web.py                 # Web 入口
├── run_worker.py              # 异步索引 worker
├── requirements.txt           # 运行与测试依赖
├── .env.example               # 配置模板
├── data/                      # 示例企业资料
├── src/qiye_rag/
│   ├── rag/                   # 加载、切分、版本索引、混合检索
│   ├── graph/                 # Agent 节点、路由与会话恢复
│   ├── web/                   # API、认证与静态页面
│   ├── run_time/              # 队列、存储、账号与 ORM 运行时
│   ├── models/                # SQLAlchemy 表模型
│   ├── evaluation/            # 评测与调优代码
│   ├── telemetry.py           # 阶段事件和 API usage
│   └── setting.py             # 集中参数
└── tests/                     # 回归与可选真实服务冒烟
```

