import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from src.data_loader import load_data
from src.symbolic_reasoning import SymbolicReasoner

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Load test data
_, _, test_data = load_data("data/raw")
class_names = list(test_data.class_indices.keys())

# Load saved model
model_path = "models/continual_updated_model.h5"
if not os.path.exists(model_path):
      model_path = "models/best_model.h5"
model = load_model(model_path)
print(f"Loaded model: {model_path}")

# Initialize symbolic reasoner
reasoner = SymbolicReasoner(class_names=class_names)

# Predict
print("Predicting on test data...")
predictions = model.predict(test_data, verbose=0)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Classification Report
print("\nClassification Report:\n")
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_names))

with open("results/classification_report.json", "w", encoding="utf-8") as f:
      json.dump(report_dict, f, indent=2)
print("Saved: results/classification_report.json")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
                  xticklabels=class_names,
                  yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()
print("Saved: results/confusion_matrix.png")

# Symbolic explanation report
filepaths = getattr(test_data, "filepaths", [""] * len(y_pred))
symbolic_rows = []

for i, probs in enumerate(predictions):
      symbolic = reasoner.explain_prediction(probs)
      true_idx = int(y_true[i])
      pred_idx = int(y_pred[i])

      symbolic_rows.append(
            {
                  "index": i,
                  "file_path": filepaths[i] if i < len(filepaths) else "",
                  "true_class": class_names[true_idx],
                  "predicted_class": class_names[pred_idx],
                  "predicted_confidence": symbolic["primary_confidence"],
                  "secondary_class": symbolic["secondary_class"],
                  "secondary_confidence": symbolic["secondary_confidence"],
                  "confidence_band": symbolic["confidence_band"],
                  "rules_fired": symbolic["rules_fired"],
                  "if_then_reasoning": symbolic["if_then_reasoning"],
                  "recommendation": symbolic["recommendation"],
                  "raw_probabilities": {
                        class_names[j]: round(float(probs[j]), 6) for j in range(len(class_names))
                  },
            }
      )

with open("results/symbolic_explanations.json", "w", encoding="utf-8") as f:
      json.dump(symbolic_rows, f, indent=2)
print("Saved: results/symbolic_explanations.json")

with open("results/symbolic_explanations.csv", "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(
            [
                  "index",
                  "file_path",
                  "true_class",
                  "predicted_class",
                  "predicted_confidence",
                  "secondary_class",
                  "secondary_confidence",
                  "confidence_band",
                  "rules_fired",
                  "if_then_reasoning",
                  "recommendation",
            ]
      )
      for row in symbolic_rows:
            writer.writerow(
                  [
                        row["index"],
                        row["file_path"],
                        row["true_class"],
                        row["predicted_class"],
                        row["predicted_confidence"],
                        row["secondary_class"],
                        row["secondary_confidence"],
                        row["confidence_band"],
                        " | ".join(row["rules_fired"]),
                        " || ".join(row["if_then_reasoning"]),
                        row["recommendation"],
                  ]
            )
print("Saved: results/symbolic_explanations.csv")

print("\nEvaluation + symbolic reasoning export complete.")

