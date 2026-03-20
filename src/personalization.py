import sqlite3
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
                    confidence_band TEXT,
                    recommendation TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add_prediction(
        self,
        patient_id: str,
        image_path: str,
        predicted_disease: str,
        prediction_confidence: float,
        confidence_band: str,
        recommendation: str,
    ) -> int:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO patient_predictions (
                    patient_id, image_path, predicted_disease, prediction_confidence,
                    confidence_band, recommendation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    image_path,
                    predicted_disease,
                    float(prediction_confidence),
                    confidence_band,
                    recommendation,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def get_recent_predictions(self, patient_id: str, limit: int = 10) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, patient_id, image_path, predicted_disease, prediction_confidence,
                       confidence_band, recommendation, created_at
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
                    "confidence_band": r[5],
                    "recommendation": r[6],
                    "created_at": r[7],
                }
            )
        return history

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
