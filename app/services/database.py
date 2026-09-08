"""SQLite persistence layer for the resume screening application.

The service is intentionally dependency-free (sqlite3 from the Python
standard library) so it can be replaced by PostgreSQL later without changing
the API contract.

Stored data:
- jobs / job descriptions
- candidates / uploaded resume metadata
- screening results
- recruiter decisions and notes
- audit events

The database is created automatically on first use.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_DB_PATH = Path("data/app.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Database:
    """Small, safe SQLite repository for screening persistence."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row

        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create/update the application schema safely."""

        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL DEFAULT '',
                    raw_text TEXT NOT NULL DEFAULT '',
                    parsed_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT,
                    file_type TEXT,
                    resume_path TEXT,
                    anonymized_text TEXT NOT NULL DEFAULT '',
                    profile_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS screening_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    candidate_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS screening_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    final_score REAL NOT NULL DEFAULT 0,
                    recommendation TEXT NOT NULL DEFAULT 'REVIEW',
                    ml_prediction INTEGER,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    recruiter_decision TEXT,
                    recruiter_notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (run_id, candidate_id),
                    FOREIGN KEY (run_id) REFERENCES screening_runs(run_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    run_id TEXT,
                    candidate_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                    ON jobs(created_at);

                CREATE INDEX IF NOT EXISTS idx_candidates_created_at
                    ON candidates(created_at);

                CREATE INDEX IF NOT EXISTS idx_screening_results_run
                    ON screening_results(run_id);

                CREATE INDEX IF NOT EXISTS idx_screening_results_candidate
                    ON screening_results(candidate_id);

                CREATE INDEX IF NOT EXISTS idx_audit_events_run
                    ON audit_events(run_id);

                CREATE INDEX IF NOT EXISTS idx_audit_events_candidate
                    ON audit_events(candidate_id);
                """
            )

    def save_job(
        self,
        job_id: str,
        title: str,
        raw_text: str,
        parsed: dict[str, Any],
    ) -> None:
        now = _utc_now()

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, title, raw_text, parsed_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    title = excluded.title,
                    raw_text = excluded.raw_text,
                    parsed_json = excluded.parsed_json
                """,
                (
                    job_id,
                    title,
                    raw_text,
                    _json(parsed),
                    now,
                ),
            )

    def save_candidate(
        self,
        candidate_id: str,
        original_filename: str,
        stored_filename: str | None,
        file_type: str | None,
        resume_path: str | None,
        anonymized_text: str,
        profile: dict[str, Any],
    ) -> None:
        now = _utc_now()

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO candidates (
                    candidate_id,
                    original_filename,
                    stored_filename,
                    file_type,
                    resume_path,
                    anonymized_text,
                    profile_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    original_filename = excluded.original_filename,
                    stored_filename = excluded.stored_filename,
                    file_type = excluded.file_type,
                    resume_path = excluded.resume_path,
                    anonymized_text = excluded.anonymized_text,
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (
                    candidate_id,
                    original_filename,
                    stored_filename,
                    file_type,
                    resume_path,
                    anonymized_text,
                    _json(profile),
                    now,
                    now,
                ),
            )

    def create_screening_run(
        self,
        run_id: str,
        job_id: str,
        candidate_count: int = 0,
        status: str = "completed",
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO screening_runs (
                    run_id,
                    job_id,
                    status,
                    candidate_count,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    job_id,
                    status,
                    max(0, int(candidate_count)),
                    _utc_now(),
                ),
            )

    def update_screening_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        candidate_count: int | None = None,
    ) -> None:
        updates: list[str] = []
        values: list[Any] = []

        if status is not None:
            updates.append("status = ?")
            values.append(status)

        if candidate_count is not None:
            updates.append("candidate_count = ?")
            values.append(max(0, int(candidate_count)))

        if not updates:
            return

        values.append(run_id)

        with self.connection() as connection:
            connection.execute(
                f"""
                UPDATE screening_runs
                SET {", ".join(updates)}
                WHERE run_id = ?
                """,
                values,
            )

    def save_screening_result(
        self,
        run_id: str,
        candidate_id: str,
        final_score: float,
        recommendation: str,
        result: dict[str, Any],
        ml_prediction: int | None = None,
    ) -> None:
        now = _utc_now()

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO screening_results (
                    run_id,
                    candidate_id,
                    final_score,
                    recommendation,
                    ml_prediction,
                    result_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                    final_score = excluded.final_score,
                    recommendation = excluded.recommendation,
                    ml_prediction = excluded.ml_prediction,
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    candidate_id,
                    float(final_score),
                    recommendation,
                    ml_prediction,
                    _json(result),
                    now,
                    now,
                ),
            )

    def save_recruiter_decision(
        self,
        run_id: str,
        candidate_id: str,
        decision: str,
        notes: str = "",
    ) -> None:
        allowed = {"SHORTLIST", "REJECT", "REVIEW", "PENDING"}

        normalized = str(decision).strip().upper()
        if normalized not in allowed:
            raise ValueError(
                f"Invalid recruiter decision: {decision!r}. "
                f"Expected one of {sorted(allowed)}."
            )

        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE screening_results
                SET recruiter_decision = ?,
                    recruiter_notes = ?,
                    updated_at = ?
                WHERE run_id = ? AND candidate_id = ?
                """,
                (
                    normalized,
                    notes or "",
                    _utc_now(),
                    run_id,
                    candidate_id,
                ),
            )

            if cursor.rowcount == 0:
                raise LookupError(
                    "Screening result not found for the supplied "
                    "run_id and candidate_id."
                )

    def add_audit_event(
        self,
        event_id: str,
        event_type: str,
        *,
        run_id: str | None = None,
        candidate_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id,
                    event_type,
                    run_id,
                    candidate_id,
                    details_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    run_id,
                    candidate_id,
                    _json(details or {}),
                    _utc_now(),
                ),
            )

    def get_screening_results(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    sr.*,
                    c.original_filename,
                    c.profile_json
                FROM screening_results sr
                JOIN candidates c
                    ON c.candidate_id = sr.candidate_id
                WHERE sr.run_id = ?
                ORDER BY sr.final_score DESC, sr.created_at ASC
                """,
                (run_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []

        for row in rows:
            item = dict(row)

            try:
                item["result"] = json.loads(item.pop("result_json"))
            except (TypeError, json.JSONDecodeError):
                item["result"] = {}

            try:
                item["candidate_profile"] = json.loads(
                    item.pop("profile_json")
                )
            except (TypeError, json.JSONDecodeError):
                item["candidate_profile"] = {}

            results.append(item)

        return results

    def get_latest_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM screening_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]


database = Database()
