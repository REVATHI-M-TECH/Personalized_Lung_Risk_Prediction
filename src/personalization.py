import sqlite3
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class PersonalizationStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    image_path TEXT,
                    predicted_disease TEXT NOT NULL,
                    prediction_confidence REAL NOT NULL,
                    secondary_disease TEXT,
                    secondary_confidence REAL,
                    confidence_band TEXT,
                    recommendation TEXT,
                    rules_fired TEXT,
                    if_then_reasoning TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Lightweight migration for existing DBs.
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(patient_predictions)").fetchall()
            }
            migration_cols = {
                "secondary_disease": "TEXT",
                "secondary_confidence": "REAL",
                "rules_fired": "TEXT",
                "if_then_reasoning": "TEXT",
            }
            for col_name, col_type in migration_cols.items():
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE patient_predictions ADD COLUMN {col_name} {col_type}"
                    )

    def get_next_patient_id(self) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT patient_id FROM patient_predictions"
            ).fetchall()

        max_num = 0
        for row in rows:
            pid = row[0] or ""
            match = re.match(r"^P-(\d+)$", pid)
            if match:
                max_num = max(max_num, int(match.group(1)))

        return f"P-{max_num + 1:04d}"

    def search_patients(self, query: str = "", limit: int = 10) -> List[Dict[str, object]]:
        q = (query or "").strip().lower()
        with self._connect() as conn:
            if q:
                rows = conn.execute(
                    """
                    SELECT patient_id, MAX(created_at) AS last_seen, COUNT(*) AS total_uploads
                    FROM patient_predictions
                    WHERE LOWER(patient_id) LIKE ?
                    GROUP BY patient_id
                    ORDER BY last_seen DESC
                    LIMIT ?
                    """,
                    (f"%{q}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT patient_id, MAX(created_at) AS last_seen, COUNT(*) AS total_uploads
                    FROM patient_predictions
                    GROUP BY patient_id
                    ORDER BY last_seen DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [
            {
                "patient_id": r[0],
                "last_seen": r[1],
                "total_uploads": int(r[2]),
            }
            for r in rows
        ]

    def add_prediction(
        self,
        patient_id: str,
        image_path: str,
        predicted_disease: str,
        prediction_confidence: float,
        secondary_disease: str,
        secondary_confidence: float,
        confidence_band: str,
        recommendation: str,
        rules_fired: List[str],
        if_then_reasoning: List[str],
    ) -> int:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO patient_predictions (
                    patient_id, image_path, predicted_disease, prediction_confidence,
                    secondary_disease, secondary_confidence,
                    confidence_band, recommendation, rules_fired, if_then_reasoning, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    image_path,
                    predicted_disease,
                    float(prediction_confidence),
                    secondary_disease,
                    float(secondary_confidence),
                    confidence_band,
                    recommendation,
                    json.dumps(rules_fired),
                    json.dumps(if_then_reasoning),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def get_recent_predictions(self, patient_id: str, limit: int = 10) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, patient_id, image_path, predicted_disease, prediction_confidence,
                      secondary_disease, secondary_confidence,
                      confidence_band, recommendation, rules_fired, if_then_reasoning, created_at
                FROM patient_predictions
                WHERE patient_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()

        history = []
        for r in rows:
            history.append(
                {
                    "id": r[0],
                    "patient_id": r[1],
                    "image_path": r[2],
                    "predicted_disease": r[3],
                    "prediction_confidence": round(float(r[4]), 4),
                    "secondary_disease": r[5],
                    "secondary_confidence": round(float(r[6] or 0.0), 4),
                    "confidence_band": r[7],
                    "recommendation": r[8],
                    "rules_fired": json.loads(r[9] or "[]"),
                    "if_then_reasoning": json.loads(r[10] or "[]"),
                    "created_at": r[11],
                }
            )
        return history

    def get_prediction_by_id(self, record_id: int) -> Dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, patient_id, image_path, predicted_disease, prediction_confidence,
                       secondary_disease, secondary_confidence,
                       confidence_band, recommendation, rules_fired, if_then_reasoning, created_at
                FROM patient_predictions
                WHERE id = ?
                """,
                (record_id,),
            ).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "patient_id": row[1],
            "image_path": row[2],
            "predicted_disease": row[3],
            "prediction_confidence": round(float(row[4]), 4),
            "secondary_disease": row[5],
            "secondary_confidence": round(float(row[6] or 0.0), 4),
            "confidence_band": row[7],
            "recommendation": row[8],
            "rules_fired": json.loads(row[9] or "[]"),
            "if_then_reasoning": json.loads(row[10] or "[]"),
            "created_at": row[11],
        }

    def get_patient_summary(self, patient_id: str) -> Dict[str, object]:
        history = self.get_recent_predictions(patient_id, limit=10)
        if not history:
            return {
                "patient_id": patient_id,
                "total_predictions": 0,
                "latest_disease": None,
                "latest_confidence": None,
                "trend": "no_history",
            }

        latest = history[0]
        total = len(history)

        if total >= 2:
            delta = latest["prediction_confidence"] - history[1]["prediction_confidence"]
            if delta > 0.05:
                trend = "confidence_improving"
            elif delta < -0.05:
                trend = "confidence_drop"
            else:
                trend = "confidence_stable"
        else:
            trend = "insufficient_history"

        return {
            "patient_id": patient_id,
            "total_predictions": total,
            "latest_disease": latest["predicted_disease"],
            "latest_confidence": latest["prediction_confidence"],
            "trend": trend,
        }
