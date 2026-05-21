"""SQLite database layer — all persistence goes through here."""
import sqlite3

from config import DB_PATH


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS channels (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                url             TEXT NOT NULL,
                language        TEXT DEFAULT 'fr',
                active          INTEGER DEFAULT 1,
                total_items     INTEGER DEFAULT 0,
                last_captured   TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS news_items (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id      TEXT NOT NULL,
                channel_name    TEXT NOT NULL,
                captured_at     TEXT NOT NULL,
                text            TEXT NOT NULL,
                summary         TEXT,
                topic           TEXT,
                language        TEXT,
                avg_confidence  REAL,
                word_count      INTEGER,
                created_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (channel_id) REFERENCES channels(id)
            );

            CREATE TABLE IF NOT EXISTS entities (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                news_item_id    INTEGER NOT NULL,
                text            TEXT NOT NULL,
                label           TEXT NOT NULL,
                FOREIGN KEY (news_item_id) REFERENCES news_items(id)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword         TEXT NOT NULL UNIQUE,
                active          INTEGER DEFAULT 1,
                hit_count       INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS alert_hits (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id        INTEGER NOT NULL,
                news_item_id    INTEGER NOT NULL,
                hit_at          TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (alert_id)     REFERENCES alerts(id),
                FOREIGN KEY (news_item_id) REFERENCES news_items(id)
            );

            CREATE INDEX IF NOT EXISTS idx_news_captured ON news_items(captured_at DESC);
            CREATE INDEX IF NOT EXISTS idx_news_channel  ON news_items(channel_id);
            CREATE INDEX IF NOT EXISTS idx_news_topic    ON news_items(topic);
            CREATE INDEX IF NOT EXISTS idx_ent_label     ON entities(label);
            CREATE INDEX IF NOT EXISTS idx_ent_text      ON entities(text);
        """)


def upsert_channel(ch: dict) -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO channels (id, name, url, language, active)
            VALUES (:id, :name, :url, :language, :active)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, url=excluded.url,
                language=excluded.language, active=excluded.active
        """, ch)


def insert_news(
    channel_id: str, channel_name: str, captured_at: str,
    text: str, summary: str | None, topic: str | None,
    language: str | None, avg_confidence: float,
    entities: list[dict],
) -> int:
    word_count = len(text.split())
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO news_items
                (channel_id, channel_name, captured_at, text, summary,
                 topic, language, avg_confidence, word_count)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (channel_id, channel_name, captured_at, text, summary,
              topic, language, avg_confidence, word_count))
        nid = cur.lastrowid

        if entities:
            conn.executemany(
                "INSERT INTO entities (news_item_id, text, label) VALUES (?,?,?)",
                [(nid, e["text"], e["label"]) for e in entities],
            )

        conn.execute("""
            UPDATE channels SET total_items=total_items+1, last_captured=?
            WHERE id=?
        """, (captured_at, channel_id))

    return nid


def get_recent_news(
    limit: int = 50,
    channel_id: str | None = None,
    topic: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM news_items WHERE 1=1"
    params: list = []
    if channel_id:
        sql += " AND channel_id=?"; params.append(channel_id)
    if topic:
        sql += " AND topic=?"; params.append(topic)
    sql += " ORDER BY captured_at DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def search_news(query: str, limit: int = 30) -> list[dict]:
    like = f"%{query}%"
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM news_items
            WHERE text LIKE ? OR summary LIKE ?
            ORDER BY captured_at DESC LIMIT ?
        """, (like, like, limit)).fetchall()
    return [dict(r) for r in rows]


def get_top_entities(
    label: str | None = None,
    limit: int = 20,
    since_hours: int = 24,
) -> list[dict]:
    sql = """
        SELECT e.text, e.label, COUNT(*) AS count
        FROM entities e
        JOIN news_items n ON e.news_item_id = n.id
        WHERE n.created_at >= datetime('now', ?)
    """
    params: list = [f"-{since_hours} hours"]
    if label:
        sql += " AND e.label=?"; params.append(label)
    sql += " GROUP BY e.text, e.label ORDER BY count DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_stats() -> dict:
    with _connect() as conn:
        total  = conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
        today  = conn.execute("SELECT COUNT(*) FROM news_items WHERE date(captured_at)=date('now')").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM channels WHERE active=1").fetchone()[0]
        topics = conn.execute("SELECT topic, COUNT(*) c FROM news_items WHERE topic IS NOT NULL GROUP BY topic ORDER BY c DESC").fetchall()
        chans  = conn.execute("SELECT id, name, total_items, last_captured FROM channels ORDER BY name").fetchall()
    return {
        "total": total, "today": today, "active_channels": active,
        "topics": [dict(r) for r in topics],
        "channels": [dict(r) for r in chans],
    }


def get_channels() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM channels ORDER BY name").fetchall()]


def get_active_alerts() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM alerts WHERE active=1").fetchall()]


def add_alert(keyword: str) -> None:
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO alerts (keyword) VALUES (?)", (keyword.lower(),))


def delete_alert(alert_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))


def record_alert_hit(alert_id: int, news_item_id: int) -> None:
    with _connect() as conn:
        conn.execute("INSERT INTO alert_hits (alert_id, news_item_id) VALUES (?,?)", (alert_id, news_item_id))
        conn.execute("UPDATE alerts SET hit_count=hit_count+1 WHERE id=?", (alert_id,))


def get_recent_alert_hits(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT ah.hit_at, a.keyword, n.channel_name, n.text, n.id AS news_id
            FROM alert_hits ah
            JOIN alerts a ON ah.alert_id = a.id
            JOIN news_items n ON ah.news_item_id = n.id
            ORDER BY ah.hit_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]
