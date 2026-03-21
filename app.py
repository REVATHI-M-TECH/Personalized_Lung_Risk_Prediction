import os
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

from src.personalization import PersonalizationStore
from src.symbolic_reasoning import SymbolicReasoner, predict_uploaded_xray_with_reasoning

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "results" / "patient_history.db"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_UPLOAD_MB = 12

MODEL_CANDIDATES = [
    BASE_DIR / "models" / "continual_updated_model.h5",
    BASE_DIR / "models" / "active_learning_subset_model.keras",
    BASE_DIR / "models" / "best_model.h5",
]


def _load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return load_model(str(model_path)), str(model_path)
    raise FileNotFoundError("No trained model file found in models/ directory")


def _discover_class_names():
    train_dir = BASE_DIR / "data" / "raw" / "train"
    if train_dir.exists():
        class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if class_names:
            return class_names

    return [
        "Bacterial Pneumonia",
        "Corona Virus Disease",
        "Normal",
        "Tuberculosis",
        "Viral Pneumonia",
    ]


model, model_path_used = _load_model()
class_names = _discover_class_names()
reasoner = SymbolicReasoner(class_names=class_names)
store = PersonalizationStore(DB_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template(
        "index.html",
        class_names=class_names,
        model_path=model_path_used,
        max_upload_mb=MAX_UPLOAD_MB,
        allowed_extensions=", ".join(sorted(ALLOWED_EXTENSIONS)),
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": model_path_used,
            "num_classes": len(class_names),
            "class_names": class_names,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    patient_id = (request.form.get("patient_id") or "").strip()
    if not patient_id:
        patient_id = store.get_next_patient_id()

    if "file" not in request.files:
        return jsonify({"error": "No file part found. Use form field name 'file'."}), 400

    file = request.files["file"]

    if file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "No selected file."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }
            ),
            400,
        )

    safe_name = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{patient_id}_{timestamp}_{safe_name}"
    upload_path = UPLOAD_DIR / safe_name
    file.save(upload_path)

    try:
        result = predict_uploaded_xray_with_reasoning(
            model=model,
            image_path=str(upload_path),
            class_names=class_names,
            image_size=(160, 160),
        )

        record_id = store.add_prediction(
            patient_id=patient_id,
            image_path=safe_name,
            predicted_disease=result["predicted_disease"],
            prediction_confidence=float(result["prediction_confidence"]),
            secondary_disease=result["secondary_disease"],
            secondary_confidence=float(result["secondary_confidence"]),
            confidence_band=result["symbolic_explanation"]["confidence_band"],
            recommendation=result["symbolic_explanation"]["recommendation"],
            rules_fired=result["symbolic_explanation"]["rules_fired"],
            if_then_reasoning=result["symbolic_explanation"]["if_then_reasoning"],
        )

        patient_summary = store.get_patient_summary(patient_id)
        patient_history = store.get_recent_predictions(patient_id, limit=5)
        for row in patient_history:
            if row.get("image_path"):
                row["image_url"] = f"/uploads/{row['image_path']}"

        if patient_summary["trend"] == "confidence_drop":
            result["symbolic_explanation"]["if_then_reasoning"].append(
                "IF patient confidence trend is dropping across visits THEN recommend urgent clinical follow-up."
            )
            result["symbolic_explanation"]["rules_fired"].append("patient_trend_confidence_drop")
        elif patient_summary["trend"] == "confidence_improving":
            result["symbolic_explanation"]["if_then_reasoning"].append(
                "IF patient confidence trend is improving across visits THEN continue current monitoring plan."
            )
            result["symbolic_explanation"]["rules_fired"].append("patient_trend_confidence_improving")

        return jsonify(
            {
                "record_id": record_id,
                "patient_id": patient_id,
                "predicted_disease": result["predicted_disease"],
                "prediction_confidence": result["prediction_confidence"],
                "secondary_disease": result["secondary_disease"],
                "secondary_confidence": result["secondary_confidence"],
                "symbolic_explanation": result["symbolic_explanation"],
                "patient_summary": patient_summary,
                "patient_history": patient_history,
                "report_json_url": f"/patient/{patient_id}/report.json",
                "report_print_url": f"/patient/{patient_id}/report/print",
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/patient/<patient_id>/history", methods=["GET"])
def patient_history(patient_id):
    limit = request.args.get("limit", default=10, type=int)
    if limit < 1 or limit > 100:
        limit = 10

    summary = store.get_patient_summary(patient_id)
    history = store.get_recent_predictions(patient_id, limit=limit)
    for row in history:
        if row.get("image_path"):
            row["image_url"] = f"/uploads/{row['image_path']}"
    return jsonify({"patient_summary": summary, "history": history})


@app.route("/patient/new-id", methods=["GET"])
def next_patient_id():
    return jsonify({"patient_id": store.get_next_patient_id()})


@app.route("/patients/search", methods=["GET"])
def search_patients():
    query = request.args.get("q", default="", type=str)
    limit = request.args.get("limit", default=10, type=int)
    if limit < 1 or limit > 100:
        limit = 10
    results = store.search_patients(query=query, limit=limit)
    return jsonify({"results": results})


@app.route("/prediction/<int:record_id>", methods=["GET"])
def prediction_detail(record_id):
    row = store.get_prediction_by_id(record_id)
    if not row:
        return jsonify({"error": "Record not found"}), 404

    if row.get("image_path"):
        row["image_url"] = f"/uploads/{row['image_path']}"
    return jsonify(row)


@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/patient/<patient_id>/report.json", methods=["GET"])
def patient_report_json(patient_id):
    summary = store.get_patient_summary(patient_id)
    history = store.get_recent_predictions(patient_id, limit=100)
    payload = {
        "patient_summary": summary,
        "history": history,
        "generated_by": "lungdp-symbolic-ui",
    }

    json_body = json.dumps(payload, indent=2)
    return Response(
        json_body,
        mimetype="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={patient_id}_report.json"
        },
    )


@app.route("/patient/<patient_id>/report/print", methods=["GET"])
def patient_report_print(patient_id):
    summary = store.get_patient_summary(patient_id)
    history = store.get_recent_predictions(patient_id, limit=20)
    return render_template(
        "patient_report.html",
        patient_id=patient_id,
        summary=summary,
        history=history,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
