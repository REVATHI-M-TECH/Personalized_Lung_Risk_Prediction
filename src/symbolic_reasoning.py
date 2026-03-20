import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from tensorflow.keras.utils import img_to_array, load_img


@dataclass
class SymbolicThresholds:
    high_confidence: float = 0.80
    medium_confidence: float = 0.60
    ambiguity_margin: float = 0.10
    low_confidence_alert: float = 0.50


class SymbolicReasoner:
    """Rule-based symbolic layer on top of neural model probabilities."""

    def __init__(self, class_names: Sequence[str], thresholds: SymbolicThresholds | None = None):
        self.class_names = list(class_names)
        if len(self.class_names) < 2:
            raise ValueError("SymbolicReasoner needs at least two classes.")
        self.thresholds = thresholds or SymbolicThresholds()
        self.disease_reason_bank = {
            "Bacterial Pneumonia": [
                "model pattern suggests focal air-space opacity that is commonly associated with bacterial consolidation",
                "high bacterial class probability compared with other classes",
            ],
            "Viral Pneumonia": [
                "model pattern suggests diffuse bilateral involvement that is often associated with viral infection",
                "viral class remains stronger than other pneumonia subclasses",
            ],
            "Tuberculosis": [
                "model pattern aligns with upper-zone dominant abnormal texture often linked with tuberculosis",
                "tuberculosis probability exceeds competing classes",
            ],
            "Corona Virus Disease": [
                "model pattern shows peripheral and bilateral abnormality tendency often linked with COVID-like findings",
                "covid class confidence is stronger than nearest alternative",
            ],
            "Normal": [
                "model found no dominant pathological class above normal baseline",
                "normal class probability is highest among all classes",
            ],
        }

    def _class_specific_reasons(self, top_label: str) -> List[str]:
        return self.disease_reason_bank.get(
            top_label,
            [
                "predicted class has highest model probability",
                "probability separation supports this class over alternatives",
            ],
        )

    def explain_prediction(self, probabilities: Sequence[float]) -> Dict[str, object]:
        probs = np.asarray(probabilities, dtype=np.float32)
        if probs.ndim != 1:
            raise ValueError("Expected 1D probability vector.")
        if len(probs) != len(self.class_names):
            raise ValueError("Probability vector length must match number of classes.")

        total = float(np.sum(probs))
        if total <= 0.0:
            raise ValueError("Probability vector sum must be positive.")

        probs = probs / total
        ranked_idx = np.argsort(probs)[::-1]
        top_idx = int(ranked_idx[0])
        second_idx = int(ranked_idx[1])

        top_conf = float(probs[top_idx])
        second_conf = float(probs[second_idx])
        margin = top_conf - second_conf
        uncertainty = 1.0 - top_conf

        top_label = self.class_names[top_idx]
        second_label = self.class_names[second_idx]

        rules_fired: List[str] = []
        reasoning_trace: List[str] = []
        recommendation = ""
        confidence_band = ""

        # IF-THEN Rule 1: high confidence and clear margin.
        if top_conf >= self.thresholds.high_confidence and margin >= self.thresholds.ambiguity_margin:
            confidence_band = "high"
            rules_fired.append("high_confidence_primary_diagnosis")
            reasoning_trace.append(
                "IF max probability >= high_confidence AND (top1 - top2) >= ambiguity_margin "
                f"THEN accept {top_label} as primary diagnosis."
            )
            recommendation = (
                f"Prediction strongly supports {top_label}. Continue with disease-specific confirmatory protocol."
            )
        # IF-THEN Rule 2: moderate confidence.
        elif top_conf >= self.thresholds.medium_confidence:
            confidence_band = "medium"
            rules_fired.append("moderate_confidence_with_differential")
            reasoning_trace.append(
                "IF max probability >= medium_confidence AND < high_confidence "
                f"THEN set {top_label} as primary with {second_label} as differential diagnosis."
            )
            recommendation = (
                f"Primary diagnosis is {top_label} with differential {second_label}. "
                "Recommend clinical correlation and targeted follow-up review."
            )
        # IF-THEN Rule 3: low confidence safety gate.
        else:
            confidence_band = "low"
            rules_fired.append("low_confidence_needs_manual_review")
            reasoning_trace.append(
                "IF max probability < medium_confidence "
                "THEN mark prediction as low-confidence and escalate to expert review."
            )
            recommendation = (
                "Low-confidence prediction. Escalate to radiologist review and consider additional tests."
            )

        # IF-THEN Rule 4: ambiguity check between top-2 classes.
        if margin < self.thresholds.ambiguity_margin:
            rules_fired.append("ambiguous_top2_overlap")
            reasoning_trace.append(
                "IF (top1 - top2) < ambiguity_margin "
                "THEN treat case as ambiguous and report top-2 differential."
            )
            recommendation += (
                f" Top-2 classes are close ({top_label} vs {second_label}); treat as uncertain differential."
            )

        # IF-THEN Rule 5: low max probability alert.
        if top_conf < self.thresholds.low_confidence_alert:
            rules_fired.append("safety_alert_low_max_probability")
            reasoning_trace.append(
                "IF max probability < low_confidence_alert "
                "THEN trigger safety alert and require manual confirmation."
            )

        class_reasons = self._class_specific_reasons(top_label)
        for reason in class_reasons:
            reasoning_trace.append(
                f"IF predicted class is {top_label} THEN supporting reason: {reason}."
            )

        return {
            "primary_class": top_label,
            "secondary_class": second_label,
            "primary_confidence": round(top_conf, 4),
            "secondary_confidence": round(second_conf, 4),
            "confidence_margin": round(margin, 4),
            "uncertainty": round(uncertainty, 4),
            "confidence_band": confidence_band,
            "rules_fired": rules_fired,
            "if_then_reasoning": reasoning_trace,
            "recommendation": recommendation,
        }

    def explain_batch(self, probabilities_batch: Sequence[Sequence[float]]) -> List[Dict[str, object]]:
        return [self.explain_prediction(row) for row in probabilities_batch]


def predict_uploaded_xray_with_reasoning(
    model,
    image_path: str,
    class_names: Sequence[str],
    image_size: tuple[int, int] = (160, 160),
    thresholds: SymbolicThresholds | None = None,
) -> Dict[str, object]:
    """Predict uploaded X-ray image and return disease + symbolic if-then explanation."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image).astype("float32") / 255.0
    input_batch = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(input_batch, verbose=0)[0]

    reasoner = SymbolicReasoner(class_names=class_names, thresholds=thresholds)
    symbolic_output = reasoner.explain_prediction(probabilities)

    return {
        "image_path": image_path,
        "predicted_disease": symbolic_output["primary_class"],
        "prediction_confidence": symbolic_output["primary_confidence"],
        "secondary_disease": symbolic_output["secondary_class"],
        "secondary_confidence": symbolic_output["secondary_confidence"],
        "symbolic_explanation": symbolic_output,
    }
