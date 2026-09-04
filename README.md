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
| 🔍 **Image OCR** | Instant text extraction from photos, documents, and screenshots using Google Gemini multimodal vision. |
| 🎙️ **Voice & Audio Transcription** | Transcribes Telegram voice notes and uploaded audio files (`.mp3`, `.wav`, `.m4a`, `.ogg`) into text. |
| 🎵 **Audio to Voice Note** | Converts standard MP3/audio files into native Telegram Opus voice message bubbles. |
| 🎛️ **Live Admin Controls (`/admin`)** | Full in-app control panel for maintenance mode, feature toggles, performance tuning, and CRM user lookups. |
| 📢 **Broadcast Engine** | Instant broadcasts, scheduled announcements (Phnom Penh UTC+7), daily recurrence, and template libraries. |
| ⚡ **Zero-Disk Streaming** | In-memory FFmpeg Opus encoding (`pipe:1`) with VoIP low-latency compression and SHA-256 LRU audio caching. |
| 🛡️ **Security & Privacy** | Rate limiting, brute-force protection, admin fallback lockout prevention, and complete `/deleteme` GDPR purge. |

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
Log in to your Anajak Cloud VPS and run:
```bash
git clone https://github.com/Heng-zm/bot-voice.git
cd bot-voice
chmod +x anajak-deploy.sh
./anajak-deploy.sh
```
*The script automatically provisions Docker, FFmpeg, creates `.env`, and launches both Bot Voice and Redis 7 in background containers.*

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
    User([👤 Telegram User]) <-->|Long Polling| TG[🤖 Telegram Bot Engine]
    
    subgraph "Core Domain Services"
        TG --> Router[🚦 Security Guard & Router]
        Router --> TTS[🗣️ TTS & Voice Pipeline]
        Router --> OCR[🔍 Vision & OCR Service]
        Router --> STT[🎙️ Transcribe Engine]
        Router --> Admin[🎛️ Admin & Broadcast Scheduler]
    end
    
    subgraph "High-Performance Layer"
        TTS --> Cache[(💾 In-Memory Audio Cache)]
        TTS --> FFmpeg[⚡ In-Memory FFmpeg pipe:1]
        Admin --> DB[(🗄️ Supabase Cloud DB)]
        OCR --> Gemini[✨ Gemini 2.0 Flash]
        STT --> Gemini
    end
```

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
│   │   │   ├── ocr.py                # Multi-provider Vision OCR pipeline
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
│   │   │   ├── commands.py           # Command handlers (/ask, /tts, /admin, /help)
│   │   │   ├── deduplication.py      # Update deduplication & idempotency
│   │   │   ├── flow.py               # Conversational flow helpers
│   │   │   ├── guards.py             # Rate limits, flood control & cooldowns
│   │   │   ├── media.py              # Voice, photo, audio & document handlers
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
├── tests/                            # Comprehensive Automated Test Suite
│   ├── test_backend_services.py      # Core service unit tests
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
