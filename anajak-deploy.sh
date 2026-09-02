#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 🇰🇭 BOT VOICE — ANAJAK CLOUD (https://anajak.cloud/) 1-CLICK DEPLOY SCRIPT
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

echo -e "${BLUE}${BOLD}"
echo "=================================================================="
echo " 🚀 ANAJAK CLOUD — BOT VOICE AUTOMATED PRODUCTION DEPLOYMENT"
echo "=================================================================="
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Root / Sudo privileges
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}ℹ️  Running with sudo permissions...${NC}"
    SUDO="sudo"
else
    SUDO=""
fi

# 2. Update package lists and install essential prerequisites
echo -e "${BLUE}📦 Step 1/4: Installing System Dependencies (curl, git, ffmpeg, ca-certificates)...${NC}"
if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq curl git ffmpeg ca-certificates ufw
elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y -q curl git ffmpeg ca-certificates
fi

# 3. Ensure Docker & Docker Compose are installed
echo -e "${BLUE}🐳 Step 2/4: Verifying Docker Engine...${NC}"
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${YELLOW}⚙️  Docker not found. Installing Docker Engine...${NC}"
    curl -fsSL https://get.docker.com | $SUDO sh
    $SUDO systemctl enable --now docker
fi

# 4. Configure .env file
echo -e "${BLUE}🔑 Step 3/4: Configuring Environment Variables (.env)...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠️  A fresh .env file was created from .env.example.${NC}"
        echo -e "${BOLD}Please edit .env now (nano .env) with your Telegram/Supabase/Gemini keys and rerun ./anajak-deploy.sh!${NC}"
        exit 1
    else
        echo -e "${RED}❌ Error: .env.example not found.${NC}"
        exit 1
    fi
fi

# 5. Build and Deploy Full-Stack Services (Bot + Redis 7)
echo -e "${BLUE}🚀 Step 4/4: Launching Bot Voice and Redis on Anajak Cloud VPS...${NC}"
$SUDO docker compose down --remove-orphans 2>/dev/null || true
$SUDO docker compose up -d --build

echo ""
echo -e "${GREEN}${BOLD}==================================================================${NC}"
echo -e "${GREEN}${BOLD} ✅ BOT VOICE SUCCESSFULLY DEPLOYED ON ANAJAK CLOUD!${NC}"
echo -e "${GREEN}${BOLD}==================================================================${NC}"
echo -e "• Webhook & System Metrics Endpoint: ${BOLD}http://YOUR_ANAJAK_IP:8080/healthz${NC}"
echo -e "• Live Container Status:             ${BOLD}docker compose ps${NC}"
echo -e "• View Live Logs:                    ${BOLD}docker compose logs -f${NC}"
echo -e "• Restart Services:                  ${BOLD}docker compose restart${NC}"
echo -e "• Stop Services:                     ${BOLD}docker compose down${NC}"
echo ""
