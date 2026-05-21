# SNRT Speech-to-Text — News Collector

Système de transcription et collecte automatique de news depuis les streams IPTV des chaînes SNRT.

## Ce que fait le système

```
Streams IPTV (Al Aoula, Arryadia...) — 24h/7j
        ↓  capture toutes les 30s
    faster-whisper  →  transcription
        ↓
    spaCy NER       →  personnes / lieux / organisations
    Groq (gratuit)  →  résumé en 2-3 phrases
    Classification  →  politique / sport / économie...
        ↓
    SQLite          →  stockage indexé
        ↓
    Dashboard web   →  feed live + recherche + alertes
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `collector.py` | Daemon 24/7 — capture, transcription, NLP, stockage |
| `dashboard.py` | Dashboard Streamlit — feed, recherche, entités, alertes |
| `pipeline.py` | Moteur STT — faster-whisper, ffmpeg, noisereduce |
| `nlp_pipeline.py` | NLP — classification sujet, NER spaCy, résumé Groq |
| `db.py` | Base de données SQLite |
| `config.py` | Configuration centralisée (lit `.env`) |
| `app.py` | Outil standalone — transcription manuelle fichier/stream |
| `run_stt.py` | CLI — transcription d'un fichier |
| `run_stream.py` | CLI — transcription d'un stream en direct |
| `nlp_keywords.py` | CLI — recherche dans un transcript JSON |

> **`app.py`** = outil ponctuel (upload fichier ou test stream).
> **`dashboard.py`** = interface du collector serveur (news en continu).

## Stack — 100% gratuit

| Composant | Outil | Coût |
|---|---|---|
| Transcription | faster-whisper (local) | Gratuit |
| NER | spaCy fr_core_news_lg (local) | Gratuit |
| Résumé IA | Groq API llama-3.3-70b | Gratuit (14 400 req/jour) |
| Base de données | SQLite | Gratuit |
| Serveur | Oracle Cloud Always Free (4 CPU, 24 GB RAM) | Gratuit à vie |
| Interface | Streamlit | Gratuit |

## Déploiement serveur (Oracle Cloud)

```bash
# 1. Cloner
git clone https://github.com/othmanedhilou/snrt-speech-to-text-mvp.git
cd snrt-speech-to-text-mvp

# 2. Installer tout
bash setup_server.sh

# 3. Configurer les URLs des streams et la clé Groq
cp .env.example .env
nano .env

# 4. Lancer
sudo systemctl start snrt-collector
sudo systemctl start snrt-dashboard

# 5. Dashboard accessible sur http://VOTRE_IP:8501
```

## Configuration `.env`

```env
GROQ_API_KEY=your_groq_api_key      # console.groq.com — gratuit
WHISPER_MODEL=small                  # small sur CPU, medium si 24GB RAM
AL_AOULA_URL=http://...stream.m3u8
ARRYADIA_URL=http://...stream.m3u8
CHUNK_DURATION=30                    # secondes par capture
```

## Utilisation locale (sans serveur)

```powershell
# Activer l'environnement
.venv\Scripts\Activate.ps1

# Dashboard collector (si collector.py tourne)
streamlit run dashboard.py

# Outil standalone (transcription manuelle)
streamlit run app.py

# CLI — transcrire un fichier
python run_stt.py --input video.mp4 --language fr

# CLI — stream en direct
python run_stream.py --url "http://stream.m3u8" --chunks 10
```

## Prérequis système

- Python 3.10+
- ffmpeg dans le PATH
- Clé API Groq (gratuite sur console.groq.com)
- URLs des streams SNRT
