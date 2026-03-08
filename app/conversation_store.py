"""
SQLite-backed conversation storage for CASCADES Chat.

Stores conversations and messages with timestamps.
Supports export to JSONL for CASCADES sleep training.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).parent / "data" / "conversations.db"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    """Thread-safe SQLite conversation storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    flagged_memory INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at DESC);
            """)

    # ── Conversations ──────────────────────────────────────────────

    def create_conversation(self, title: str = "New Chat") -> dict:
        conv_id = str(uuid.uuid4())
        now = _utcnow()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now),
            )
        return {"id": conv_id, "title": title, "created_at": now, "updated_at": now, "pinned": False}

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, title, created_at, updated_at, pinned,
                          (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
                   FROM conversations c
                   ORDER BY pinned DESC, updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_conversation(self, conv_id: str, title: Optional[str] = None, pinned: Optional[bool] = None):
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if pinned is not None:
            updates.append("pinned = ?")
            params.append(int(pinned))
        if not updates:
            return
        updates.append("updated_at = ?")
        params.append(_utcnow())
        params.append(conv_id)
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def delete_conversation(self, conv_id: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    # ── Messages ───────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        flagged_memory: bool = False,
    ) -> dict:
        msg_id = str(uuid.uuid4())
        now = _utcnow()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO messages (id, conversation_id, role, content, created_at, flagged_memory)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, role, content, now, int(flagged_memory)),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": now,
            "flagged_memory": flagged_memory,
        }

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def flag_message(self, msg_id: str, flagged: bool = True):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE messages SET flagged_memory = ? WHERE id = ?",
                (int(flagged), msg_id),
            )

    # ── Auto Title ─────────────────────────────────────────────────

    def auto_title(self, conv_id: str) -> str:
        """Generate a conversation title from the first user message."""
        msgs = self.get_messages(conv_id)
        for m in msgs:
            if m["role"] == "user":
                text = m["content"].strip()
                title = text[:60] + ("..." if len(text) > 60 else "")
                self.update_conversation(conv_id, title=title)
                return title
        return "New Chat"

    # ── Export for Training ─────────────────────────────────────────

    def export_training_data(
        self,
        output_path: Path,
        since: Optional[str] = None,
        flagged_only: bool = False,
    ) -> int:
        """Export conversations as JSONL for CASCADES sleep training.

        Returns the number of training examples written.
        """
        with self._get_conn() as conn:
            query = "SELECT DISTINCT conversation_id FROM messages WHERE 1=1"
            params: list = []

            if since:
                query += " AND created_at > ?"
                params.append(since)
            if flagged_only:
                query += " AND flagged_memory = 1"

            conv_ids = [
                r["conversation_id"]
                for r in conn.execute(query, params).fetchall()
            ]

        count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for cid in conv_ids:
                messages = self.get_messages(cid)
                if len(messages) < 2:
                    continue

                # Build conversation pairs: user → assistant
                for i, msg in enumerate(messages):
                    if msg["role"] == "user" and i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        if next_msg["role"] == "assistant":
                            example = {
                                "prompt": msg["content"],
                                "response": next_msg["content"],
                                "conversation_id": cid,
                                "flagged": bool(
                                    msg["flagged_memory"] or next_msg["flagged_memory"]
                                ),
                                "timestamp": msg["created_at"],
                            }
                            f.write(json.dumps(example, ensure_ascii=False) + "\n")
                            count += 1

        return count

    # ── Stats ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._get_conn() as conn:
            n_convos = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            n_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_flagged = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE flagged_memory = 1"
            ).fetchone()[0]
        return {
            "conversations": n_convos,
            "messages": n_msgs,
            "flagged_memories": n_flagged,
        }
