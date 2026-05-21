"""SNRT News Collector — Streamlit dashboard.

Run:
    streamlit run dashboard.py --server.port 8501
"""
from datetime import datetime

import pandas as pd
import streamlit as st

import db
from config import DASHBOARD_REFRESH_SECONDS

st.set_page_config(
    page_title="SNRT News Collector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=DASHBOARD_REFRESH_SECONDS * 1_000, key="refresh")
except ImportError:
    pass  # manual refresh only

# ── Helpers ───────────────────────────────────────────────────────────────────
TOPIC_ICONS = {
    "politique":    "🔵 Politique",
    "sport":        "🟢 Sport",
    "economie":     "🟡 Économie",
    "societe":      "🟠 Société",
    "international":"🔴 International",
    "culture":      "🟣 Culture",
    "meteo":        "⚪ Météo",
    "faits_divers": "⚫ Faits divers",
    "general":      "⬜ Général",
}

def fmt_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return iso


def topic_label(topic: str | None) -> str:
    return TOPIC_ICONS.get(topic or "general", "⬜ Général")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 SNRT News")
    st.caption("Surveillance IPTV en temps réel")
    st.divider()

    channels = db.get_channels()
    ch_opts  = {"Toutes les chaînes": None} | {c["name"]: c["id"] for c in channels}
    sel_ch   = ch_opts[st.selectbox("Chaîne", list(ch_opts.keys()))]

    topic_opts = {"Tous les sujets": None} | {v: k for k, v in TOPIC_ICONS.items()}
    sel_topic  = topic_opts[st.selectbox("Sujet", list(topic_opts.keys()))]

    st.divider()
    st.subheader("🔔 Alertes")
    alerts = db.get_active_alerts()
    for a in alerts:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{a['keyword']}** — {a['hit_count']} hits")
        if c2.button("🗑", key=f"del_{a['id']}"):
            db.delete_alert(a["id"])
            st.rerun()

    kw = st.text_input("Nouveau mot-clé")
    if st.button("➕ Ajouter alerte") and kw.strip():
        db.add_alert(kw.strip())
        st.rerun()

    st.divider()
    st.caption(f"Rafraîchissement : {DASHBOARD_REFRESH_SECONDS}s")

# ── Header stats ──────────────────────────────────────────────────────────────
st.title("📡 SNRT News Collector")
stats = db.get_stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total news", f"{stats['total']:,}")
c2.metric("Aujourd'hui", stats["today"])
c3.metric("Chaînes actives", stats["active_channels"])
top = stats["topics"][0]["topic"] if stats["topics"] else "—"
c4.metric("Sujet dominant", topic_label(top))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_feed, tab_search, tab_entities, tab_alerts, tab_channels = st.tabs([
    "📰 Feed", "🔍 Recherche", "👤 Entités", "🔔 Alertes", "📡 Chaînes",
])

# ── Tab 1 — Feed ──────────────────────────────────────────────────────────────
with tab_feed:
    news = db.get_recent_news(limit=40, channel_id=sel_ch, topic=sel_topic)

    if not news:
        st.info("Aucune news pour ces filtres. Vérifiez que le collector tourne.")
    else:
        for item in news:
            tl = topic_label(item.get("topic"))
            ts = fmt_time(item.get("captured_at"))
            ch = item.get("channel_name", "")
            conf = item.get("avg_confidence") or 0.0
            words = item.get("word_count") or 0
            summary = item.get("summary", "")
            text    = item.get("text", "")

            header = f"{tl} &nbsp;|&nbsp; **{ch}** &nbsp;|&nbsp; {ts}"
            with st.expander(header):
                if summary:
                    st.success(f"**Résumé :** {summary}")
                st.markdown(
                    f"> {text[:600]}{'...' if len(text) > 600 else ''}"
                )
                st.caption(f"Confiance : {conf:.0%} &nbsp;|&nbsp; Mots : {words} &nbsp;|&nbsp; Langue : {item.get('language','?')}")

# ── Tab 2 — Search ────────────────────────────────────────────────────────────
with tab_search:
    q = st.text_input("Rechercher...", placeholder="ministre, Rabat, football, CAN...")
    if q.strip():
        results = db.search_news(q.strip(), limit=25)
        st.write(f"**{len(results)} résultat(s)** pour « {q} »")
        for item in results:
            tl = topic_label(item.get("topic"))
            ts = fmt_time(item.get("captured_at"))
            with st.expander(f"{tl} | {item.get('channel_name')} | {ts}"):
                if item.get("summary"):
                    st.success(item["summary"])
                # Basic bold highlight
                highlighted = item["text"].replace(q, f"**{q}**")
                st.markdown(f"> {highlighted[:700]}")

# ── Tab 3 — Entities ──────────────────────────────────────────────────────────
with tab_entities:
    hours = st.slider("Fenêtre temporelle (heures)", 1, 168, 24)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("👤 Personnes")
        pers = db.get_top_entities("PER", 20, hours)
        if pers:
            st.dataframe(
                pd.DataFrame(pers)[["text", "count"]].rename(columns={"text": "Nom", "count": "Mentions"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Pas de données")

    with col_r:
        st.subheader("📍 Lieux")
        locs = db.get_top_entities("LOC", 20, hours)
        if locs:
            st.dataframe(
                pd.DataFrame(locs)[["text", "count"]].rename(columns={"text": "Lieu", "count": "Mentions"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Pas de données")

    st.subheader("🏢 Organisations")
    orgs = db.get_top_entities("ORG", 20, hours)
    if orgs:
        st.dataframe(
            pd.DataFrame(orgs)[["text", "count"]].rename(columns={"text": "Organisation", "count": "Mentions"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Pas de données")

# ── Tab 4 — Alerts ────────────────────────────────────────────────────────────
with tab_alerts:
    st.subheader("Dernières alertes déclenchées")
    hits = db.get_recent_alert_hits(limit=30)
    if hits:
        for h in hits:
            ts = fmt_time(h.get("hit_at"))
            kw = h.get("keyword", "")
            ch = h.get("channel_name", "")
            st.warning(f"🔔 **{kw}** — {ch} — {ts}")
            st.caption(h.get("text", "")[:200])
    else:
        st.info("Aucune alerte déclenchée.")

# ── Tab 5 — Channels ──────────────────────────────────────────────────────────
with tab_channels:
    st.subheader("État des chaînes")
    chans = stats.get("channels", [])
    if chans:
        for ch in chans:
            last = fmt_time(ch.get("last_captured"))
            items = ch.get("total_items", 0)
            st.write(f"📡 **{ch['name']}** — {items} news — dernier : {last}")
    else:
        st.warning("Aucune chaîne. Configurez les URLs dans le fichier `.env` puis redémarrez le collector.")
        st.code("""
# .env
AL_AOULA_URL=http://votre-url/stream.m3u8
ARRYADIA_URL=http://votre-url/stream.m3u8
GROQ_API_KEY=votre_cle_groq
WHISPER_MODEL=small
        """)
