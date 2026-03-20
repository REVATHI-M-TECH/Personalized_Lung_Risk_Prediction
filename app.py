import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
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
        return jsonify({"error": "patient_id is required."}), 400

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
            image_path=str(upload_path),
            predicted_disease=result["predicted_disease"],
            prediction_confidence=float(result["prediction_confidence"]),
            confidence_band=result["symbolic_explanation"]["confidence_band"],
            recommendation=result["symbolic_explanation"]["recommendation"],
        )

        patient_summary = store.get_patient_summary(patient_id)
        patient_history = store.get_recent_predictions(patient_id, limit=5)

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
    return jsonify({"patient_summary": summary, "history": history})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
