# app/db.py
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timezone

Row = Tuple[int, str, str, str, Optional[float], Optional[str]]
# (faiss_id, entry_id, ts, text, sentiment, tags)


class DB:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _con(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._con() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    faiss_id  INTEGER PRIMARY KEY,
                    entry_id  TEXT    NOT NULL,
                    ts        TEXT    NOT NULL,
                    text      TEXT    NOT NULL,
                    sentiment REAL,
                    tags      TEXT
                )
                """
            )

    def insert_entry(
        self,
        faiss_id: int,
        entry_id: str,
        text: str,
        sentiment: Optional[float] = None,
        tags: Optional[str] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._con() as con:
            con.execute(
                "INSERT INTO entries (faiss_id, entry_id, ts, text, sentiment, tags) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (faiss_id, entry_id, ts, text, sentiment, tags),
            )

    def fetch_recent(self, days: int = 7) -> List[Row]:
        # days is kept for compatibility; current query returns all, newest first
        with self._con() as con:
            cur = con.execute(
                "SELECT faiss_id, entry_id, ts, text, sentiment, tags "
                "FROM entries ORDER BY ts DESC"
            )
            return cur.fetchall()

    def fetch_by_ids(self, faiss_ids: List[int]) -> List[Row]:
        if not faiss_ids:
            return []
        placeholders = ",".join(["?"] * len(faiss_ids))
        with self._con() as con:
            cur = con.execute(
                f"SELECT faiss_id, entry_id, ts, text, sentiment, tags "
                f"FROM entries WHERE faiss_id IN ({placeholders})",
                tuple(faiss_ids),
            )
            rows = cur.fetchall()
        order = {fid: i for i, fid in enumerate(faiss_ids)}
        rows.sort(key=lambda r: order.get(r[0], 1e9))
        return rows

    # Reindex helpers

    def fetch_all(self) -> List[Row]:
        with self._con() as con:
            cur = con.execute(
                "SELECT faiss_id, entry_id, ts, text, sentiment, tags "
                "FROM entries ORDER BY ts ASC"
            )
            return cur.fetchall()

    def replace_all(self, new_rows: List[Row]) -> None:
        with self._con() as con:
            cur = con.cursor()
            cur.execute("BEGIN")
            try:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entries_new (
                        faiss_id  INTEGER PRIMARY KEY,
                        entry_id  TEXT    NOT NULL,
                        ts        TEXT    NOT NULL,
                        text      TEXT    NOT NULL,
                        sentiment REAL,
                        tags      TEXT
                    )
                    """
                )
                cur.executemany(
                    "INSERT INTO entries_new (faiss_id, entry_id, ts, text, sentiment, tags) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    new_rows,
                )
                cur.execute("DROP TABLE IF EXISTS entries")
                cur.execute("ALTER TABLE entries_new RENAME TO entries")
                con.commit()
            except Exception:
                con.rollback()
                raise

    def update_faiss_id(self, entry_id: str, new_faiss_id: int) -> None:
        with self._con() as con:
            con.execute(
                "UPDATE entries SET faiss_id = ? WHERE entry_id = ?",
                (new_faiss_id, entry_id),
            )
            con.commit()
