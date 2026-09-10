<div align="center">

# 🎙️ Bot Voice
### Next-Generation Multilingual Text-to-Speech, OCR & AI Assistant for Telegram

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![Telegram Bot API](https://img.shields.io/badge/Telegram-Bot%20API-2CA5E0?logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Supabase Database](https://img.shields.io/badge/Database-Supabase-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*High-performance, low-latency voice synthesis in Khmer & 10+ languages, image OCR transcription, intelligent multimodal chat, and full-featured broadcast scheduling.*

[Features](#-key-features) • [Quickstart](#-quickstart-in-3-steps) • [Deployment](#-easy-server-deployment) • [Admin Panel](#-telegram-admin-panel-admin) • [Architecture](#-architecture) • [Project Structure](#-project-structure)

---
</div>

## ✨ Key Features

| Domain | Capabilities |
| :--- | :--- |
| 🗣️ **Text-to-Speech (TTS)** | High-speed Edge TTS & Hugging Face Kiri Space integration. Supports **Khmer**, English, Chinese, Korean, Japanese, Hindi, Malay, Indonesian, Filipino, and Arabic. |
| 📢 **Channel Auto-Voice Narrator** | Automatically synthesizes studio-grade voice note narration for Telegram channel posts (text & captions) with smart URL/delimiter cleaning, sentence-level cuts, `#notts` opt-out tags, and whitelist control. |
| 📄 **PDF & Document Voice Synthesis** | Reads `.pdf` documents uploaded by users, transcribes text using Gemini Multimodal Vision, and outputs interactive voice playback. |
| 🔍 **Image OCR** | Instant text extraction from photos, documents, and screenshots using Google Gemini multimodal vision with automatic model fallback (`gemini-2.0-flash`). |
| 🎙️ **Voice & Audio Transcription** | Transcribes Telegram voice notes and uploaded audio files (`.mp3`, `.wav`, `.m4a`, `.ogg`) into text. |
| 🎵 **Audio to Voice Note** | Converts standard MP3/audio files into native Telegram Opus voice message bubbles. |
| 🎛️ **Live Admin Controls (`/admin`)** | Full in-app control panel for maintenance mode, feature toggles, performance tuning, and CRM user lookups. |
| 📢 **Broadcast Engine** | Instant broadcasts, scheduled announcements (Phnom Penh UTC+7), daily recurrence, and bulk blocked-user persistence (99% Supabase API savings). |
| ⚡ **Zero-Disk Streaming & Caching** | In-memory FFmpeg Opus encoding (`pipe:1`) with VoIP low-latency compression and SHA-256 LRU audio caching. |
| 🛡️ **Self-Healing Infrastructure** | 4-tier resilient speech fallback (HF $\to$ Edge $\to$ Gemini $\to$ Emergency Edge), idle database reconnection recovery (`RemoteProtocolError` resilience), and rate limits. |

---

## ⚡ Quickstart in 3 Steps

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Heng-zm/bot-voice.git
cd bot-voice
python -m pip install -r requirements.txt
```

### 2. Configure Minimal Environment (Only 5 Lines!)
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
ADMIN_IDS=1272791365
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Launch the Bot
```bash
python -m app.main
# or
./start.sh
```

---

## 🚀 Easy Server Deployment

### 🐳 Option 1: Docker Compose (Recommended)
Deploy 24/7 in a background container with automated health management and log rotation:
```bash
# 1. Edit your .env file
cp .env.example .env

# 2. Build and run
docker compose up -d --build

# 3. View live output
docker compose logs -f
```

---

### ⚡ Option 2: Automated 1-Click VPS Script (`deploy.sh`)
Works on any Linux VPS (**Ubuntu 22.04 / 24.04, Debian 12, CentOS, AlmaLinux**):
```bash
chmod +x deploy.sh
./deploy.sh
```
*The script automatically checks dependencies, creates virtual environments, verifies FFmpeg, and starts the service.*

---

### ☁️ Option 3: Anajak Cloud (https://anajak.cloud/)

#### 🚀 Method 1: 1-Click Automated VPS Script (Recommended)
Log in to your Anajak Cloud VPS via SSH and run:
```bash
git clone https://github.com/Heng-zm/bot-voice.git
cd bot-voice
chmod +x anajak-deploy.sh
./anajak-deploy.sh
```
*The script automatically detects your public IP, provisions Docker Engine, configures UFW firewall, verifies container health, and launches both Bot Voice and Redis 7.*

##### 🛠️ Anajak Cloud Management Toolkit:
```bash
./anajak-deploy.sh logs      # View live streaming container logs
./anajak-deploy.sh status    # Check container health and memory usage
./anajak-deploy.sh restart   # Restart bot and redis containers
./anajak-deploy.sh test      # Run automated unit test suite inside Docker
./anajak-deploy.sh backup    # Run Supabase database backup utility
./anajak-deploy.sh update    # Pull latest Git code and rebuild containers
./anajak-deploy.sh stop      # Stop all background services
```

#### 🐳 Method 2: Manual Docker Compose
```bash
git clone https://github.com/Heng-zm/bot-voice.git
cd bot-voice
cp .env.example .env
# Edit .env with your credentials: nano .env
docker compose up -d --build
docker compose logs -f
```

---

### 🌐 Option 4: Wasmer Edge (https://wasmer.io)

Deploy globally to Wasmer Edge with zero server maintenance:
```bash
# 1. Install Wasmer CLI (if not already installed)
curl https://get.wasmer.io -sSfL | sh

# 2. Login to your Wasmer account
wasmer login

# 3. Deploy using the configured wasmer.toml
wasmer deploy
```
*Wasmer will automatically package the container, deploy to the edge network, and expose your `/healthz`, `/system`, `/tts`, and Telegram Webhook endpoints globally.*

---

### ⚙️ Option 5: Linux Systemd Daemon (Native VPS Service)
```bash
# 1. Copy service template
sudo cp bot-voice.service /etc/systemd/system/bot-voice.service

# 2. Edit paths & user
sudo nano /etc/systemd/system/bot-voice.service

# 3. Enable & Start
sudo systemctl daemon-reload
sudo systemctl enable --now bot-voice

# 4. View status & logs
sudo systemctl status bot-voice
journalctl -u bot-voice -f
```

---

## 🎛️ Telegram Admin Panel (`/admin`)

Manage every aspect of your bot dynamically from Telegram without editing `.env` or restarting servers:

```
/admin
├── ⚙️ Settings          — Toggle TTS, OCR, Voice Transcribe, AI Resolver, Maintenance Mode
├── ⚡ Performance       — Hot-reload DB Workers, Audio Cache MB, TTL, Edge Parallel Streams
├── 📢 Broadcasts        — Compose, preview, schedule (Phnom Penh UTC+7), and template manager
├── 👥 User CRM          — Search user by ID/@username, inspect preferences, block/unblock
├── 📊 Live Metrics      — Real-time memory usage, cache hit ratios, and latency graphs
└── 🚨 Error Center      — Inspect recent runtime exceptions and stack traces
```

---

## 🏗️ Architecture

```mermaid
flowchart TD
    User([👤 Telegram User / DM]) <-->|Webhook / Polling| TG[🤖 Telegram Gateway]
    Channel([📢 Public Channel Post]) -->|channel_post| TG
    
    subgraph "Ingress & Protection"
        TG --> Guard[🚦 Security Guard & Anti-Spam Shield]
        Guard --> CB[⚡ 60s Sliding-Window Circuit Breaker]
        CB --> Router{Dispatcher}
    end

    subgraph "Core Domain Services"
        Router --> TTS[🗣️ Resilient 4-Tier TTS Pipeline]
        Router --> ChanNarrator[📢 Channel Auto-Voice Narrator]
        Router --> OCR[🔍 Vision & PDF Document OCR]
        Router --> AI[🧠 AI Assistant & Translator]
        Router --> Admin[🎛️ Admin CRM & Broadcast Scheduler]
        
        ChanNarrator -->|Sanitized Text| TTS
    end
    
    subgraph "Multi-Tier Resilient TTS Engine"
        TTS --> T1[Tier 1: Hugging Face Khmer Space ≤250 chars]
        T1 -.->|Empty / Limit / Timeout <1s| T2[Tier 2: Microsoft Edge Neural TTS]
        T2 -.->|Multilingual / Retry| T3[Tier 3: Google Gemini Multimodal]
        T3 -.->|Timeout >10s / Quota| T4[Tier 4: Fast Edge Fallback]
    end

    subgraph "AI Multi-Model Fallback Chain"
        AI & OCR --> G1[Primary: Gemini 2.0 Flash]
        G1 -.->|429 Quota Exceeded| G2[Fallback: Gemini 1.5 Flash]
        G2 -.->|Failover| G3[Emergency: Gemini 2.5 Flash / HF Qwen]
    end
    
    subgraph "High-Performance Persistence & Cache"
        T1 & T2 & T3 & T4 --> Cache[(💾 In-Memory Audio Cache SHA-256)]
        T1 & T2 & T3 & T4 --> FFmpeg[⚡ In-Memory FFmpeg Opus pipe:1]
        Admin & Guard --> MemCache[(⚡ Zero-Wait In-Memory Settings Cache)]
        MemCache <--> Redis[(⚡ Redis 7 - Distributed Locks & Telemetry)]
        MemCache <--> DB[(🗄️ Supabase PostgreSQL - Pooler / Transaction Mode)]
        DB --> Pruner[🧹 Bounded Batch Pruning 500 records/batch]
    end
```

### 🌟 Architectural Highlights

1. **Zero-Wait In-Memory Cold Boot:**
   - Bot settings load and hot-apply in memory in `< 1ms` during container boot, eliminating 16 sequential database write roundtrips.
2. **Multi-Tier Khmer TTS with Sub-Second Failover:**
   - Hugging Face Space ZeroGPU handles natural Khmer voices with safe `≤ 250` character chunking. If the Space returns empty audio or character limits, the pipeline instantly drops to Microsoft Edge TTS in `< 1s` without wasting retry backoff delays.
3. **Sliding-Window Circuit Breakers:**
   - Database and AI providers are shielded with a 60-second sliding window circuit breaker. Isolated errors do not accumulate into false infrastructure outages.
   - If Supabase is restarting or temporarily paused, the bot operates **100% autonomously** in memory and Redis.
4. **Database Resource Protection & Bounded Pruning:**
   - Background maintenance jobs prune old history and cache in **bounded batches of 500 records** with dedicated indexes on `created_at DESC`, preventing table locks, memory spikes, and PostgREST timeouts.
5. **Multi-Model Quota Resilience:**
   - Vision OCR and AI chat automatically cascade across `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash` $\rightarrow$ `gemini-2.5-flash` $\rightarrow$ `Qwen2.5-7B`, preventing 429 quota disruptions.

---

## 📁 Project Structure

```
bot-voice/
├── app/                              # Core Application Codebase
│   ├── core/                         # Core Configurations & Security
│   │   ├── config.py                 # Pydantic Settings & environment variables
│   │   └── telegram_auth.py          # Admin dynamic authorization & fallback
│   ├── services/                     # Domain Services Architecture
│   │   ├── ai/                       # AI, Multimodal Vision & Embeddings
│   │   │   ├── gemini.py             # Google Gemini client & auto-fallback (2.0/1.5)
│   │   │   ├── ocr.py                # Multi-provider Vision OCR & PDF pipeline
│   │   │   ├── language.py           # Fast regex + langdetect language detection
│   │   │   ├── providers.py          # Provider interfaces & routing
│   │   │   └── vector_store.py       # Upstash / Supabase Vector similarity search
│   │   ├── broadcast/                # Mass Messaging & Scheduled Announcements
│   │   │   └── templates.py          # Broadcast layout templates & presets
│   │   ├── settings/                 # Dynamic Runtime Configuration Store
│   │   │   └── store.py              # Supabase/PostgreSQL settings key-value store
│   │   ├── telegram/                 # Modular Telegram Bot System
│   │   │   ├── buttons.py            # Inline keyboard layouts & UI builders
│   │   │   ├── callbacks.py          # Inline button callback queries & state machine
│   │   │   ├── channel.py            # Channel Auto-Voice Narrator & text cleaning
│   │   │   ├── commands.py           # Command handlers (/ask, /tts, /admin, /help)
│   │   │   ├── deduplication.py      # Update deduplication & idempotency
│   │   │   ├── flow.py               # Conversational flow helpers
│   │   │   ├── guards.py             # Rate limits, flood control & channel protection
│   │   │   ├── media.py              # Voice, photo, audio, document & PDF handlers
│   │   │   ├── routing.py            # Central Telegram handler registration
│   │   │   ├── security.py           # Admin authentication & privilege enforcement
│   │   │   └── workloads.py          # Background worker tasks & audio rendering
│   │   ├── tts/                      # Text-to-Speech Engines & Pipelines
│   │   │   ├── engine.py             # Edge TTS & Hugging Face Kiri integration
│   │   │   ├── cache.py              # In-memory SHA-256 LRU audio cache
│   │   │   └── voices.py             # Supported voice catalogs & mappings
│   │   ├── users/                    # User CRM & Preferences
│   │   │   └── prefs.py              # User language, voice, and speed persistence
│   │   └── health.py                 # Internal health probe server
│   ├── utils/                        # Shared Utilities
│   │   ├── file_io.py                # Safe temp file lifecycle & cleanup
│   │   ├── text.py                   # Khmer & multilingual text sanitization
│   │   └── time.py                   # Phnom Penh (UTC+7) timezone formatting
│   ├── bot.py                        # Telegram Bot builder & polling runner
│   ├── legacy.py                     # Legacy engine compatibility & state
│   └── main.py                       # FastAPI application & REST API endpoints
├── backups/                          # Local Database Backups (JSON & CSV, git-ignored)
├── tests/                            # Comprehensive Automated Test Suite
│   ├── test_backend_services.py      # Core service unit tests
│   ├── test_channel_narrator.py      # Channel narrator & audio text cleaning tests
│   ├── test_startup_script.py        # Startup script validation
│   ├── test_system_upgrades.py       # API & webhook regression tests
│   ├── test_telegram.py              # Telegram command & message tests
│   └── test_telegram_auth.py         # Admin authorization tests
├── ev/                               # Environment template archives
│   └── .env.example                  # Reference environment template
├── static/                           # Static assets, logos & templates
├── .dockerignore                     # Docker build exclusions
├── .env.example                      # Root configuration template
├── .gitignore                        # Git exclusion rules
├── Dockerfile                        # Multi-stage production container image
├── Procfile                          # Cloud platform process file
├── anajak-deploy.sh                  # 1-click Anajak Cloud deployment script
├── backup_data.py                    # 1-click Standalone Supabase Data Backup Script
├── bot-voice.service                 # Linux systemd daemon definition
├── deploy.sh                         # 1-click Linux VPS automated installer
├── docker-compose.yml                # Docker Compose orchestration (Bot + Redis)
├── main.py                           # Root launcher entrypoint
├── pyproject.toml                    # Modern PEP 621 / Poetry packaging metadata
├── render.yaml                       # Render.com Blueprint configuration
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development & testing dependencies
├── run_local.ps1                     # Windows local development launcher
├── start.sh                          # Production container startup script
├── supabase_bot_setup.sql            # PostgreSQL schema, migrations & RLS
└── upload.sftp                       # SFTP direct file deployment batch script

---

## 🧪 Testing & Verification

Run the full automated test suite (41+ tests across language detection, security, replay stores, TTS caching, and user preferences):

```bash
# Run all unit tests
python -m unittest discover -s tests -v

# Run Ruff code linter
python -m ruff check .

# Syntax byte-compile check
python -m compileall -q app
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
