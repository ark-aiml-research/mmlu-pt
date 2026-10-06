"""Transactional SQLite checkpoints; called only from the single event-loop writer."""

import fcntl
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import canonical_json, timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS questions (
 annotation_id TEXT PRIMARY KEY, input_json TEXT NOT NULL, candidates_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rows (
 position INTEGER PRIMARY KEY, row_index INTEGER NOT NULL, annotation_id TEXT NOT NULL REFERENCES questions
);
CREATE TABLE IF NOT EXISTS tasks (
 generation INTEGER NOT NULL, annotation_id TEXT NOT NULL REFERENCES questions,
 kind TEXT NOT NULL, pass_number INTEGER NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending', result_json TEXT, error TEXT,
 retries INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (generation, annotation_id, kind, pass_number)
);
CREATE TABLE IF NOT EXISTS attempts (
 generation INTEGER NOT NULL, annotation_id TEXT NOT NULL, kind TEXT NOT NULL, pass_number INTEGER NOT NULL,
 attempt INTEGER NOT NULL, seed INTEGER NOT NULL, result_json TEXT, error TEXT,
 model_metadata_json TEXT NOT NULL, duration_seconds REAL NOT NULL, timestamp TEXT NOT NULL,
 request_status TEXT NOT NULL, started_at TEXT NOT NULL,
 PRIMARY KEY (generation, annotation_id, kind, pass_number, attempt)
);
"""


@contextmanager
def run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another process is using {run_dir}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Checkpoint:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('generation', '1')")

    def close(self) -> None:
        self.connection.close()

    @property
    def generation(self) -> int:
        return int(self.connection.execute("SELECT value FROM metadata WHERE key='generation'").fetchone()[0])

    def new_generation(self) -> int:
        with self.connection:
            self.connection.execute("UPDATE metadata SET value=? WHERE key='generation'", (str(self.generation + 1),))
        return self.generation

    def prepare(self, rows: list[dict], passes: int) -> None:
        now = timestamp()
        with self.connection:
            for row in rows:
                self.connection.execute("INSERT OR IGNORE INTO questions VALUES (?, ?, ?)",
                                        (row["annotation_id"], canonical_json(row["input"]), canonical_json(row["candidates"])))
                self.connection.execute("INSERT OR IGNORE INTO rows VALUES (?, ?, ?)",
                                        (row["position"], row["row_index"], row["annotation_id"]))
                for number in range(1, passes + 1):
                    self.connection.execute(
                        "INSERT OR IGNORE INTO tasks (generation,annotation_id,kind,pass_number,created_at,updated_at) VALUES (?,?, 'independent',?,?,?)",
                        (self.generation, row["annotation_id"], number, now, now))
            self.connection.execute("UPDATE tasks SET status='pending',updated_at=? WHERE generation=? AND status IN ('running','error')",
                                    (now, self.generation))
            self.connection.execute("UPDATE attempts SET request_status='interrupted',error='interrupted_request',timestamp=? "
                                    "WHERE generation=? AND request_status='running'", (now, self.generation))

    def ensure_adjudication(self, annotation_id: str) -> None:
        now = timestamp()
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO tasks (generation,annotation_id,kind,pass_number,created_at,updated_at) VALUES (?,?,'adjudication',0,?,?)",
                (self.generation, annotation_id, now, now))

    def pending(self, kind: str, pass_number: int):
        return self.connection.execute(
            "SELECT t.*,q.input_json,q.candidates_json FROM tasks t JOIN questions q USING(annotation_id) "
            "WHERE t.generation=? AND t.kind=? AND t.pass_number=? AND t.status!='success' ORDER BY t.annotation_id",
            (self.generation, kind, pass_number))

    def tasks(self, annotation_id: str) -> list[dict]:
        rows = self.connection.execute("SELECT * FROM tasks WHERE generation=? AND annotation_id=? ORDER BY kind DESC,pass_number",
                                       (self.generation, annotation_id))
        return [dict(row) | {"result": json.loads(row["result_json"]) if row["result_json"] else None} for row in rows]

    def mark_running(self, task: dict) -> int:
        key = (self.generation, task["annotation_id"], task["kind"], task["pass_number"])
        with self.connection:
            self.connection.execute("UPDATE tasks SET status='running',updated_at=? WHERE generation=? AND annotation_id=? AND kind=? AND pass_number=?",
                                    (timestamp(), *key))
        return self.connection.execute(
            "SELECT COALESCE(MAX(attempt),0) FROM attempts WHERE generation=? AND annotation_id=? AND kind=? AND pass_number=?", key).fetchone()[0]

    def start_attempt(self, task: dict, attempt: int, seed: int, model_metadata: dict) -> None:
        now = timestamp()
        with self.connection:
            self.connection.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                    (self.generation, task["annotation_id"], task["kind"], task["pass_number"],
                                     attempt, seed, None, None, canonical_json(model_metadata), 0.0, now, "running", now))

    def persist_attempt(self, task: dict, attempt: dict) -> None:
        now = timestamp()
        result = canonical_json(attempt["result"]) if attempt["result"] is not None else None
        key = (self.generation, task["annotation_id"], task["kind"], task["pass_number"], attempt["attempt"])
        original = self.connection.execute("SELECT model_metadata_json FROM attempts WHERE generation=? AND annotation_id=? "
                                           "AND kind=? AND pass_number=? AND attempt=?", key).fetchone()
        if original is None:
            raise ValueError("Attempt was not durably registered before its request")
        metadata = json.loads(original[0]) | attempt["metadata"]
        with self.connection:
            self.connection.execute("UPDATE attempts SET result_json=?,error=?,model_metadata_json=?,duration_seconds=?,"
                                    "timestamp=?,request_status=? WHERE generation=? AND annotation_id=? AND kind=? AND pass_number=? AND attempt=?",
                                    (result, attempt["error"], canonical_json(metadata), attempt["duration_seconds"], now,
                                     "success" if result else "error", *key))
            self.connection.execute(
                "UPDATE tasks SET status=?,result_json=?,error=?,retries=?,updated_at=? WHERE generation=? AND annotation_id=? AND kind=? AND pass_number=?",
                ("success" if result else "running", result, attempt["error"], attempt["attempt"] - 1, now,
                 self.generation, task["annotation_id"], task["kind"], task["pass_number"]))

    def finish_error(self, task: dict, error: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE tasks SET status='error',error=?,updated_at=? WHERE generation=? AND annotation_id=? AND kind=? AND pass_number=?",
                (error, timestamp(), self.generation, task["annotation_id"], task["kind"], task["pass_number"]))

    def question_ids(self) -> list[str]:
        return [row[0] for row in self.connection.execute("SELECT annotation_id FROM questions ORDER BY annotation_id")]

    def row_mapping(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM rows ORDER BY position")]
