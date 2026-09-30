import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.core.config import SESSIONS_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    plan TEXT NOT NULL,
    current_exercise TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    exercise TEXT,
    frame_ts REAL,
    fase TEXT,
    desplazamiento_y REAL,
    postura_correcta INTEGER,
    hombros_visibles INTEGER,
    repeticiones INTEGER,
    tiempo_total_s REAL DEFAULT 0,
    tiempo_valido_s REAL DEFAULT 0,
    objetivo_tiempo_s REAL DEFAULT 0,
    cumplimiento_postural REAL DEFAULT 0,
    correcciones INTEGER DEFAULT 0,
    angulo_corporal_deg REAL DEFAULT 0,
    error_postural_deg REAL DEFAULT 0,
    level TEXT,
    message_es TEXT,
    siguiente_paso TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_observations_session
ON observations (session_id);
"""

OBSERVATION_COLUMNS = {
    "tiempo_total_s": "REAL DEFAULT 0",
    "tiempo_valido_s": "REAL DEFAULT 0",
    "objetivo_tiempo_s": "REAL DEFAULT 0",
    "cumplimiento_postural": "REAL DEFAULT 0",
    "correcciones": "INTEGER DEFAULT 0",
    "angulo_corporal_deg": "REAL DEFAULT 0",
    "error_postural_deg": "REAL DEFAULT 0",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, db_path=SESSIONS_DB_PATH):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)
            existing = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(observations)").fetchall()
            }
            for name, ddl in OBSERVATION_COLUMNS.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE observations ADD COLUMN {name} {ddl}")

    def create_session(self, plan: dict) -> dict:
        session_id = uuid.uuid4().hex
        created_at = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, status, plan, created_at) VALUES (?, 'active', ?, ?)",
                (session_id, json.dumps(plan, ensure_ascii=False), created_at),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def set_current_exercise(self, session_id: str, exercise: str):
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET current_exercise = ? WHERE id = ?",
                (exercise, session_id),
            )

    def complete_session(self, session_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET status = 'completed', completed_at = ? WHERE id = ? AND status = 'active'",
                (_now(), session_id),
            )
        return cur.rowcount > 0

    def abandon_session(self, session_id: str) -> bool:
        """Marca la sesión como 'abandoned' (terminada a medias o por error)."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET status = 'abandoned', completed_at = ? WHERE id = ? AND status = 'active'",
                (_now(), session_id),
            )
        return cur.rowcount > 0

    def add_observation(self, session_id: str, obs: dict, correction: dict):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observations (
                    session_id, exercise, frame_ts, fase, desplazamiento_y,
                    postura_correcta, hombros_visibles, repeticiones,
                    tiempo_total_s, tiempo_valido_s, objetivo_tiempo_s,
                    cumplimiento_postural, correcciones, angulo_corporal_deg,
                    error_postural_deg, level, message_es, siguiente_paso, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    obs.get("exercise"),
                    obs.get("frame_ts"),
                    obs.get("fase"),
                    obs.get("desplazamiento_y"),
                    int(obs.get("postura_correcta", True)),
                    int(obs.get("hombros_visibles", False)),
                    obs.get("repeticiones"),
                    obs.get("tiempo_total_s", 0.0),
                    obs.get("tiempo_valido_s", 0.0),
                    obs.get("objetivo_tiempo_s", 0.0),
                    obs.get("cumplimiento_postural", 0.0),
                    obs.get("correcciones", 0),
                    obs.get("angulo_corporal_deg", 0.0),
                    obs.get("error_postural_deg", 0.0),
                    correction.get("level"),
                    correction.get("message_es"),
                    correction.get("siguiente_paso"),
                    _now(),
                ),
            )

    def list_observations(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM observations WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _row_to_session(self, row: sqlite3.Row) -> dict:
        return {
            "session_id": row["id"],
            "status": row["status"],
            "plan": json.loads(row["plan"]),
            "current_exercise": row["current_exercise"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }


_store = None


def get_store(db_path=SESSIONS_DB_PATH) -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore(db_path=db_path)
    return _store
