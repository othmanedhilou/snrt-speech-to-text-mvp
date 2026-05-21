#!/usr/bin/env bash
# SNRT News Collector — Oracle Cloud Ubuntu server setup
# Run once: bash setup_server.sh

set -e

echo "======================================================"
echo " SNRT News Collector — Server Setup"
echo "======================================================"

# ── System packages ───────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip ffmpeg git

# ── Python virtual environment ────────────────────────────
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ── spaCy French model ────────────────────────────────────
python -m spacy download fr_core_news_lg

# ── Directory structure ───────────────────────────────────
mkdir -p data outputs logs outputs/chunks

# ── Environment file ──────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo ">>> .env created. Edit it and add your channel URLs and GROQ_API_KEY."
    echo ">>> nano .env"
fi

# ── Systemd service: collector ────────────────────────────
WORK_DIR=$(pwd)
VENV_PYTHON="$WORK_DIR/.venv/bin/python"
USER=$(whoami)

sudo tee /etc/systemd/system/snrt-collector.service > /dev/null <<EOF
[Unit]
Description=SNRT News Collector daemon
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
ExecStart=$VENV_PYTHON collector.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ── Systemd service: dashboard ────────────────────────────
sudo tee /etc/systemd/system/snrt-dashboard.service > /dev/null <<EOF
[Unit]
Description=SNRT News Dashboard (Streamlit)
After=network.target snrt-collector.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
ExecStart=$WORK_DIR/.venv/bin/streamlit run dashboard.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable snrt-collector snrt-dashboard

echo ""
echo "======================================================"
echo " Setup complete!"
echo "======================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your channel URLs and Groq API key:"
echo "     nano .env"
echo ""
echo "  2. Start services:"
echo "     sudo systemctl start snrt-collector"
echo "     sudo systemctl start snrt-dashboard"
echo ""
echo "  3. Check logs:"
echo "     sudo journalctl -fu snrt-collector"
echo "     sudo journalctl -fu snrt-dashboard"
echo ""
echo "  4. Dashboard accessible at:"
echo "     http://YOUR_SERVER_IP:8501"
echo ""
echo "  5. Open port 8501 in Oracle Cloud security rules:"
echo "     Cloud Console > Networking > VCN > Security List > Add Ingress Rule"
echo "     Source: 0.0.0.0/0 | Port: 8501 | Protocol: TCP"
echo "======================================================"
