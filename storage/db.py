"""
storage/db.py
-------------
Lightweight SQLite repository for storing validation and backtest run logs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime


class RunDatabase:
    """Manages persistence of validation reports in SQLite."""

    def __init__(self, db_path: str = "backtestguard.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_passed INTEGER NOT NULL,
                    has_lookahead_risk INTEGER NOT NULL,
                    total_return_pct REAL,
                    sharpe_ratio REAL,
                    report_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_run(self, report_dict: dict) -> int:
        """Insert validation report and return inserted run ID."""
        metrics = report_dict.get("metrics") or {}
        tot_ret = metrics.get("total_return_pct")
        sharpe = metrics.get("sharpe_ratio")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (timestamp, dataset_passed, has_lookahead_risk, total_return_pct, sharpe_ratio, report_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report_dict.get("timestamp", datetime.now().isoformat()),
                    1 if report_dict.get("dataset_passed") else 0,
                    1 if report_dict.get("has_lookahead_risk") else 0,
                    tot_ret,
                    sharpe,
                    json.dumps(report_dict),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def list_runs(self, limit: int = 20) -> list[dict]:
        """Fetch historical runs."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, dataset_passed, has_lookahead_risk, total_return_pct, sharpe_ratio
                FROM runs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_run(self, run_id: int) -> dict | None:
        """Fetch full report by run ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT report_json FROM runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["report_json"])
            return None
