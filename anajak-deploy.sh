#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 🇰🇭 BOT VOICE — ANAJAK CLOUD (https://anajak.cloud/) ALL-IN-ONE PRODUCTION TOOLKIT
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Root / Sudo privileges
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Subcommand handling
ACTION="${1:-deploy}"

case "$ACTION" in
    logs)
        echo -e "${BLUE}📊 Viewing live container logs...${NC}"
        $SUDO docker compose logs -f
        exit 0
        ;;
    status)
        echo -e "${BLUE}📊 Checking Anajak Cloud container status...${NC}"
        $SUDO docker compose ps
        exit 0
        ;;
    restart)
        echo -e "${YELLOW}♻️ Restarting Bot Voice and Redis containers...${NC}"
        $SUDO docker compose restart
        echo -e "${GREEN}✅ Restart complete.${NC}"
        exit 0
        ;;
    stop|down)
        echo -e "${RED}🛑 Stopping Anajak Cloud containers...${NC}"
        $SUDO docker compose down
        echo -e "${GREEN}✅ Stopped.${NC}"
        exit 0
        ;;
    test)
        echo -e "${CYAN}🧪 Running automated unit test suite inside Docker...${NC}"
        $SUDO docker compose run --rm bot-voice python -m unittest discover -s tests -p "test_*.py"
        exit 0
        ;;
    backup)
        echo -e "${CYAN}💾 Running database backup utility...${NC}"
        $SUDO docker compose run --rm bot-voice python backup_data.py
        exit 0
        ;;
    update)
        echo -e "${BLUE}🔄 Pulling latest updates from Git...${NC}"
        git pull || echo -e "${YELLOW}Git pull skipped or not a git repo.${NC}"
        echo -e "${BLUE}🚀 Rebuilding containers with latest code...${NC}"
        $SUDO docker compose up -d --build
        echo -e "${GREEN}✅ Update complete!${NC}"
        exit 0
        ;;
    deploy)
        # Continue to full deployment sequence below
        ;;
    help|--help|-h)
        echo -e "${BOLD}Anajak Cloud VPS Management Script${NC}"
        echo "Usage: ./anajak-deploy.sh [command]"
        echo ""
        echo "Commands:"
        echo "  deploy   (default) Provision dependencies, build, and start containers"
        echo "  logs     View live streaming container logs"
        echo "  status   Show container status and resource usage"
        echo "  restart  Restart all bot containers"
        echo "  stop     Stop and remove containers"
        echo "  test     Run unit test suite inside container"
        echo "  backup   Run Supabase database backup utility"
        echo "  update   Pull latest Git commits and rebuild"
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown command: $ACTION${NC}"
        echo "Run './anajak-deploy.sh help' for usage."
        exit 1
        ;;
esac

echo -e "${BLUE}${BOLD}"
echo "=================================================================="
echo " 🚀 ANAJAK CLOUD — BOT VOICE AUTOMATED PRODUCTION DEPLOYMENT"
echo "=================================================================="
echo -e "${NC}"

# 1. Auto-detect Server Public IP
echo -e "${CYAN}🌐 Detecting Anajak Cloud VPS Public IP...${NC}"
SERVER_IP=$(curl -s4 --max-time 3 https://api.ipify.org 2>/dev/null || curl -s4 --max-time 3 https://ifconfig.me 2>/dev/null || curl -s4 --max-time 3 https://icanhazip.com 2>/dev/null || echo "YOUR_SERVER_IP")
echo -e "   Detected Server IP: ${BOLD}${SERVER_IP}${NC}"

# 2. Update package lists and install essential prerequisites
echo -e "${BLUE}📦 Step 1/4: Installing System Dependencies (curl, git, ffmpeg, ca-certificates)...${NC}"
if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq curl git ffmpeg ca-certificates ufw
elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y -q curl git ffmpeg ca-certificates
fi

# 3. Configure UFW firewall if active
if command -v ufw >/dev/null 2>&1 && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
    echo -e "${YELLOW}🛡️ Configuring UFW Firewall for Anajak Cloud VPS...${NC}"
    $SUDO ufw allow 22/tcp comment 'SSH' >/dev/null 2>&1 || true
    $SUDO ufw allow 8080/tcp comment 'Bot Voice' >/dev/null 2>&1 || true
fi

# 4. Ensure Docker & Docker Compose are installed
echo -e "${BLUE}🐳 Step 2/4: Verifying Docker Engine & Compose...${NC}"
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${YELLOW}⚙️  Docker not found. Installing Docker Engine...${NC}"
    curl -fsSL https://get.docker.com | $SUDO sh
    $SUDO systemctl enable --now docker
fi

# 5. Configure .env file
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

# Ensure Anajak environment variables are set in .env if not present
if ! grep -q "ANAJAK_URL=" .env 2>/dev/null; then
    echo "" >> .env
    echo "# Anajak Cloud VPS Auto Configuration" >> .env
    echo "ANAJAK_URL=http://${SERVER_IP}:8080" >> .env
    echo "ANAJAK_PUBLIC_URL=http://${SERVER_IP}:8080" >> .env
fi

# 6. Build and Deploy Full-Stack Services (Bot + Redis 7)
echo -e "${BLUE}🚀 Step 4/4: Launching Bot Voice and Redis on Anajak Cloud VPS...${NC}"
$SUDO docker compose down --remove-orphans 2>/dev/null || true
$SUDO docker compose up -d --build

# 7. Health Check Verification
echo -e "${CYAN}🏥 Verifying container health on http://127.0.0.1:8080/healthz...${NC}"
HEALTH_OK=false
for i in {1..15}; do
    if curl -sf http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
        HEALTH_OK=true
        break
    fi
    sleep 2
done

echo ""
echo -e "${GREEN}${BOLD}==================================================================${NC}"
if [ "$HEALTH_OK" = true ]; then
    echo -e "${GREEN}${BOLD} ✅ BOT VOICE SUCCESSFULLY DEPLOYED & HEALTHY ON ANAJAK CLOUD!${NC}"
else
    echo -e "${YELLOW}${BOLD} ⚠️ BOT VOICE CONTAINERS STARTED (Service warming up...)${NC}"
fi
echo -e "${GREEN}${BOLD}==================================================================${NC}"
echo -e "• Healthz Endpoint:             ${BOLD}http://${SERVER_IP}:8080/healthz${NC}"
echo -e "• System Telemetry Endpoint:     ${BOLD}http://${SERVER_IP}:8080/system${NC}"
echo -e "• AI Assistant Endpoint:        ${BOLD}http://${SERVER_IP}:8080/ai-assistant${NC}"
echo -e "• TTS Voice Synthesis Endpoint: ${BOLD}http://${SERVER_IP}:8080/tts${NC}"
echo ""
echo -e "${BOLD}Management Commands:${NC}"
echo -e "• View live streaming logs:     ${CYAN}./anajak-deploy.sh logs${NC}"
echo -e "• Check container status:       ${CYAN}./anajak-deploy.sh status${NC}"
echo -e "• Restart services:             ${CYAN}./anajak-deploy.sh restart${NC}"
echo -e "• Run unit test suite:          ${CYAN}./anajak-deploy.sh test${NC}"
echo -e "• Run database backup:          ${CYAN}./anajak-deploy.sh backup${NC}"
echo -e "• Pull updates and rebuild:     ${CYAN}./anajak-deploy.sh update${NC}"
echo -e "• Stop all services:            ${CYAN}./anajak-deploy.sh stop${NC}"
echo ""
