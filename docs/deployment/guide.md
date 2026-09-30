# Deployment & Operations Guide

## 1. Quick Start with Docker Compose
The fastest and most reliable way to run Bot Voice in production:

```bash
cp .env.example .env
# Edit .env with your TELEGRAM_BOT_TOKEN and other credentials

docker compose up -d --build
```

View live logs:
```bash
docker compose logs -f bot-voice
```

## 2. Linux VPS Native Deployment (Systemd)
For running directly on Ubuntu/Debian/RHEL VPS:

```bash
chmod +x ./deploy/deploy.sh
./deploy/deploy.sh --systemd
```

Manage the systemd service:
```bash
sudo systemctl status bot-voice
sudo systemctl restart bot-voice
sudo journalctl -u bot-voice -f
```

## 3. Maintenance Scripts
- **Database Backup**: `python scripts/backup.py`
- **Database Restore**: `python scripts/restore.py`
- **Database Seeding**: `python scripts/seed.py`
- **System Maintenance & Temp File Sweep**: `python scripts/maintenance.py`
