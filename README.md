# 🤖 Personal Assistant — Feishu-Optimized AI Agent Framework

A lightweight, LLM-powered personal AI assistant with deep Feishu (Lark) integration, multi-channel support, long-term memory, context compression, and CardKit streaming messages.

---

## Features at a Glance

- **Multi-channel messaging** — Feishu, Telegram, Discord, and WhatsApp via a unified message bus
- **ReAct agent loop** — async reasoning with tool calling, subagent isolation, and token budget tracking
- **Personal long-term memory** — SQLite + BM25 with multi-factor ranking (preference, decision, reference, constraint, profile)
- **Session context compression** — automatic rolling summarization to prevent context overflow
- **Feishu CardKit streaming** — typewriter-style token streaming with live tool logs and token usage charts
- **Progressive Skills** — Markdown-based skill packs with 3-level lazy loading
- **Built-in security** — dangerous command blocking, path traversal protection, workspace sandboxing

---

## Project Structure

```
personal-assistant-feishu/
├── assistant/                      # Core framework
│   ├── agent/                    # 🧠 Agent engine
│   │   ├── loop.py              #   ReAct main loop
│   │   ├── context.py           #   Prompt building & context management
│   │   ├── memory.py           #   File-based persistent memory
│   │   ├── memory_compiler.py  #   LLM-assisted memory extraction & merging
│   │   ├── memory_retriever.py #   Personal memory retrieval & injection
│   │   ├── personal_memory_store.py  # SQLite long-term storage (BM25 retrieval)
│   │   ├── skills.py           #   Skills loader
│   │   ├── subagent.py        #   Background subagent manager
│   │   └── tools/             #   🛠️ Built-in tool set
│   │       ├── filesystem.py   #     File read/write/edit
│   │       ├── shell.py       #     Shell command execution (safety-filtered)
│   │       ├── web.py         #     Web search & scraping
│   │       ├── message.py     #     Cross-channel message sending
│   │       ├── notion.py      #     Notion database management
│   │       ├── pdf_mineru.py #     MinerU PDF parsing
│   │       ├── image_generate.py  #  Image generation + Feishu delivery
│   │       ├── session_manage.py  #  Multi-session management
│   │       ├── spawn.py       #     Subagent spawning
│   │       ├── cron.py        #     Scheduled task management
│   │       ├── memory_search.py  #  Personal memory search
│   │       ├── transcription.py  #  Speech-to-text (Groq / iFlytek)
│   │       └── meeting_summary.py #  Meeting summary generation
│   ├── channels/               # 📡 Chat channel adapters
│   │   ├── feishu.py          #   ⭐ Feishu WebSocket long connection
│   │   ├── telegram.py        #   Telegram Bot
│   │   ├── discord.py         #   Discord Bot
│   │   └── whatsapp.py        #   WhatsApp Bridge
│   ├── session/               # 💬 Session management
│   │   ├── manager.py         #   Session persistence (JSONL) + multi-session switching
│   │   └── compressor.py      #   Rolling summarization (context overflow prevention)
│   ├── bus/                   # 🚌 Message bus
│   │   ├── queue.py          #   Async message queue
│   │   └── events.py         #   Message event definitions
│   ├── providers/             # 🤖 LLM providers
│   │   └── litellm_provider.py  #  LiteLLM unified wrapper
│   ├── cron/                  # ⏰ Scheduled task service
│   ├── heartbeat/             # 💓 Heartbeat service (periodic agent wake-up)
│   ├── config/               # ⚙️  Configuration (Pydantic)
│   ├── cli/                  # 🖥️  CLI commands
│   ├── skills/               # 📦 Built-in Skills packages
│   │   ├── github/          #   GitHub CLI integration
│   │   ├── weather/          #   Weather lookup
│   │   ├── tmux/            #   Remote tmux control
│   │   ├── summarize/        #   URL/YouTube/file summarization
│   │   ├── cron/            #   Scheduled reminders
│   │   └── skill-creator/  #   Skill creation assistant
│   └── utils/               # 🧰 Utilities
├── workspace/                 # 📁 Agent workspace template
│   ├── AGENTS.md            #   Agent behavior instructions
│   ├── SOUL.md              #   Agent persona definition
│   ├── TOOLS.md             #   Tool usage guide
│   ├── HEARTBEAT.md         #   Periodic task list
│   └── USER.md              #   User profile
├── tests/                     # 🧪 Test suite
├── bridge/                    # 🔌 WhatsApp Bridge (TypeScript)
├── ui/                       # 🌐 Web UI (optional, Streamlit)
├── scripts/                  # 🛠️ Helper scripts
├── config.example.json       # 📋 Configuration template
└── pyproject.toml           # 📦 Project dependencies
```

---

## Core Technical Highlights

### 1. 🧠 ReAct Agent Loop

The core engine is an async ReAct loop: **receive message → build context → call LLM → if LLM requests a tool, execute it → inject result → repeat until done → return response**.

| Feature | Detail |
|---------|--------|
| 🔁 Iteration cap | Prevents infinite loops (default: 20 per turn) |
| 📊 Token budget tracking | Real-time input/output/cache token accounting |
| 🤖 Subagent isolation | Complex background tasks run independently, report back on completion |

### 2. 🧬 Personal Long-Term Memory

A hybrid retrieval system built on **SQLite + BM25** with multi-dimensional weighted scoring:

| Dimension | Description |
|-----------|-------------|
| Memory type | preference / decision / reference / constraint / profile |
| Scope | global / topic / project |
| BM25 full-text | Joint scoring across keywords, summary, and content |
| Priority | User-marked important memories get boosted weight |
| Recency | Recently accessed memories rank higher |

**Workflow**: end of each conversation turn → MemoryCompiler extracts candidate memories → LLM-assisted merge decision → sync to SQLite + `MEMORY.md`

### 3. 📉 Session Context Compression

Keeps long conversations within the LLM context window:

- **Trigger**: message count exceeds threshold (default: 80) or estimated tokens exceed threshold (default: 12,000)
- **Strategy**: keep last 25 messages + 8 tool messages; compress the rest into a rolling summary
- **No gaps**: the boundary between the summary window and recent messages is seamless

### 4. 📡 Feishu CardKit Streaming

Real-time streaming messages via Feishu CardKit OpenAPI:

| Feature | Description |
|---------|-------------|
| ⌨️ Typewriter effect | Content streamed token-by-token to a Feishu card |
| 📊 Token chart | Live ECharts stacked bar showing token usage |
| 🔧 Tool logs | Collapsible panel with real-time tool execution status |
| ⚡ Preemptive timeout | Graceful downgrade to plain text before CardKit hard timeout |

### 5. 📚 Progressive Skills Loading

Skills are Markdown-format capability packs with three-tier lazy loading to control context overhead:

```
Level 1 (always in context):  name + description       (~100 words)
Level 2 (loaded on demand):  SKILL.md body            (~<5000 words)
Level 3 (read on demand):    scripts/ + references/    (unlimited — execute, don't read)
```

Built-in Skills: 🐙 GitHub, 🌤️ Weather, 🧵 tmux Control, 🧾 Summarizer, ⏰ Reminders

### 6. 🔐 Tool Security

- **Shell execution**: blocks dangerous patterns (`rm -rf`, `mkfs`, etc.), workspace directory restrictions, path traversal detection
- **File access**: optional `restrict_to_workspace` confines all file ops to the workspace
- **Cron tasks**: supports one-time reminders and recurring jobs, delivered via the message bus

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| LLM wrapper | LiteLLM (OpenRouter / OpenAI / Anthropic / DeepSeek / Gemini / Zhipu / Moonshot / vLLM) |
| Feishu SDK | lark-oapi |
| WebSocket | websockets |
| Config | Pydantic + pydantic-settings |
| Scheduling | croniter |
| Logging | loguru |
| CLI | Typer + Rich |
| PDF parsing | MinerU API |
| Speech-to-text | Groq Whisper / iFlytek WebAPI |
| Build | Hatchling |

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd personal-assistant-feishu
pip install -e .
```

### 2. Initialize

```bash
personal-assistant onboard
```

### 3. Configure

The config file defaults to `~/.personal-assistant/config.json`. Override with an environment variable:

```bash
export PERSONAL_ASSISTANT_HOME=/path/to/.personal-assistant
```

Copy `config.example.json` to `~/.personal-assistant/config.json` and fill in the required API keys.

### 4. Configure the Feishu Bot

1. Go to [Feishu Open Platform](https://open.feishu.cn/app) → Create app → enable **Bot** capability
2. **Permissions**: add the following scopes
   - `im:message` / `im:message:send_as_bot`
   - `im:resource` / `im:message:readonly`
   - `im:message.p2p_msg:readonly`
   - `cardkit:card:write`
3. **Events**: subscribe to `im.message.receive_v1`, select **long connection** mode (no public IP required)
4. Get **App ID** and **App Secret** from "Credentials & Basic Info"
5. **Card template**: create a card template in Feishu Card Builder, copy the template ID
6. Publish the app

### 5. Launch

```
personal-assistant gateway       # 🚀 Start gateway (Feishu Bot + Cron)
personal-assistant agent -m "Hi"  # 💬 Single-shot conversation
personal-assistant agent          # 🗣️  Interactive chat mode
personal-assistant onboard        # ⚙️  Initialize configuration
```

---

## Configuration Reference

### Full config.json Example

API keys are redacted. Copy `config.example.json` to `~/.personal-assistant/config.json` and fill in real values.

```json
{
  "agents": {
    "defaults": {
      "workspace": "$PERSONAL_ASSISTANT_HOME/workspace",
      "model": "anthropic/claude-sonnet-4-6-thinking",
      "maxTokens": 10240,
      "tokenBudgetMode": "output",
      "mergeSubagentUsage": true,
      "temperature": 0.7,
      "maxToolIterations": 50
    }
  },
  "providers": {
    "openrouter": { "apiKey": "YOUR_OPENROUTER_API_KEY", "apiBase": null },
    "openai": { "apiKey": "YOUR_OPENAI_API_KEY", "apiBase": null }
  },
  "channels": {
    "feishu": {
      "enabled": true,
      "appId": "YOUR_FEISHU_APP_ID",
      "appSecret": "YOUR_FEISHU_APP_SECRET",
      "cardTemplateId": "YOUR_CARD_TEMPLATE_ID",
      "streamingEnabled": true,
      "streamingPreemptiveTimeoutSec": 480
    }
  },
  "tools": {
    "web": { "search": { "apiKey": "YOUR_SERPER_API_KEY", "maxResults": 5 } },
    "mineru": { "enabled": true, "token": "YOUR_MINERU_TOKEN" },
    "imageGen": { "enabled": true, "apiKey": "YOUR_IMAGE_GEN_API_KEY" },
    "notion": { "enabled": true, "apiKey": "YOUR_NOTION_API_KEY" },
    "contextCompression": { "enabled": true, "triggerByMessageCount": 80 },
    "memorySystem": { "enabled": true }
  }
}
```

### Supported LLM Providers

| Provider | Notes | Get API Key |
|----------|-------|-------------|
| `openrouter` | Recommended; supports all major models | [openrouter.ai](https://openrouter.ai) |
| `anthropic` | Direct Claude access | [console.anthropic.com](https://console.anthropic.com) |
| `openai` | GPT / o-series | [platform.openai.com](https://platform.openai.com) |
| `deepseek` | Direct DeepSeek access | [platform.deepseek.com](https://platform.deepseek.com) |
| `gemini` | Direct Gemini access | [aistudio.google.com](https://aistudio.google.com) |
| `groq` | LLM + Whisper transcription | [console.groq.com](https://console.groq.com) |
| `zhipu` | Zhipu GLM | [open.bigmodel.cn](https://open.bigmodel.cn) |
| `moonshot` | Kimi/Moonshot | [platform.moonshot.cn](https://platform.moonshot.cn) |
| `vllm` | Self-hosted vLLM | — |

### Tool Configuration

| Tool | Config Path | Required Fields |
|------|-------------|-----------------|
| Web search | `tools.web.search` | `apiKey` (Serper) |
| MinerU PDF | `tools.mineru` | `token` |
| Image generation | `tools.imageGen` | `apiBase`, `apiKey`, `modelName` |
| Notion | `tools.notion` | `apiKey`, `databaseId` |
| iFlytek transcription | `tools.iflytek` | `appId`, `secretKey` |
| Personal memory | `tools.memorySystem` | — (all optional) |
| Context compression | `tools.contextCompression` | — (all optional) |

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `personal-assistant onboard` | Initialize config and workspace |
| `personal-assistant agent -m "..."` | Single-shot message |
| `personal-assistant agent` | Interactive chat mode |
| `personal-assistant gateway` | Start gateway (Feishu Bot + Cron) |
| `personal-assistant status` | Show current status |
| `personal-assistant cron add` | Add a scheduled task |
| `personal-assistant cron list` | List scheduled tasks |
| `personal-assistant cron remove <id>` | Remove a scheduled task |

---

## Testing

```bash
pytest tests/ -v
```

Test coverage includes: Agent loop, token monitoring, session history limits, memory retrieval, Feishu streaming timeout, and tool validation.

---

*Licensed under the MIT License.*
