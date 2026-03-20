import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.data_loader import load_data

# ----------------------------
# Parameters
# ----------------------------
K = 50
EPOCHS = 2
LEARNING_RATE = 5e-5
ITERATIONS = 8
FREEZE_LAYERS = True
BATCH_SIZE_FIT = 2
BATCH_SIZE_PREDICT = 16
CONTINUAL_SAMPLES = 100
CONTINUAL_LEARNING_RATE = 1e-6  # Very conservative: 10x smaller than AL learning rate
CONTINUAL_EPOCHS = 10  # More epochs for gradual learning
TRAIN_VAL_SPLIT = 0.8  # Use 80% of train for pool, 20% for eval


def now_ms(start_time):
    return (time.perf_counter() - start_time) * 1000.0


LOG_PATH = "results/execution_pipeline.log"
stage_times_ms = {}


def pipeline_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def start_stage(stage_name):
    pipeline_log(f"START | {stage_name}")
    return time.perf_counter()


def end_stage(stage_name, stage_start):
    elapsed = now_ms(stage_start)
    stage_times_ms[stage_name] = elapsed
    pipeline_log(f"END   | {stage_name} | {elapsed:.2f} ms")

# ----------------------------
# Create results folder
# ----------------------------
os.makedirs("results", exist_ok=True)
with open(LOG_PATH, "w", encoding="utf-8") as log_file:
    log_file.write("Execution Pipeline Log\n")

pipeline_start = time.perf_counter()
pipeline_log("PIPELINE START")

# ----------------------------
# Load data generators
# ----------------------------
stage_start = start_stage("load_data_generators")
pipeline_log("Loading data generators for full 10K images")
train_gen, val_gen, test_gen = load_data("data/raw", batch_size=16)
end_stage("load_data_generators", stage_start)

num_classes = len(train_gen.class_indices)
pipeline_log(f"Number of classes: {num_classes}")
pipeline_log(f"Total training samples: {train_gen.samples}")

# ----------------------------
# Convert generators to numpy arrays (in chunks to avoid memory crash)
# ----------------------------
stage_start = start_stage("load_training_arrays")
pipeline_log("Loading training data in memory (streaming from generator)")
images, labels = [], []

# Reset generator
train_gen.reset()
for x_batch, y_batch in train_gen:
    images.append(x_batch)
    labels.append(y_batch)
    if len(images) * 16 >= train_gen.samples:  # Loaded all samples
        break

images = np.vstack(images)
labels = np.vstack(labels)

pipeline_log(f"Successfully loaded {len(images)} images")
end_stage("load_training_arrays", stage_start)

# ----------------------------
# Shuffle before split
# ----------------------------
indices = np.arange(len(images))
np.random.shuffle(indices)

images = images[indices]
labels = labels[indices]

# ----------------------------
# Split into Pool (for active learning) and Evaluation
# ----------------------------
split_index = int(TRAIN_VAL_SPLIT * len(images))

pool_images = images[:split_index]
pool_labels = labels[:split_index]

eval_images = images[split_index:]
eval_labels = labels[split_index:]

pipeline_log(f"Pool size (for active learning): {len(pool_images)}")
pipeline_log(f"Evaluation size (for testing): {len(eval_images)}")

# ----------------------------
# Load model
# ----------------------------
stage_start = start_stage("load_and_compile_model")
pipeline_log("Loading pre-trained model")
model = load_model("models/best_model.h5")

if FREEZE_LAYERS:
    for layer in model.layers[:-10]:
        layer.trainable = False
    pipeline_log("Frozen base layers for fine-tuning")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
end_stage("load_and_compile_model", stage_start)

# ----------------------------
# ACTIVE LEARNING LOOP
# ----------------------------
pipeline_log("===== Starting Active Learning Phase =====")
accuracy_list = []
f1_list = []
selected_count = 0

for iteration in range(1, ITERATIONS + 1):
    iter_start = time.perf_counter()

    pipeline_log(f"AL Iteration {iteration}/{ITERATIONS} | pool={len(pool_images)}")
    
    # Get predictions on entire pool
    pipeline_log("Computing uncertainty scores on pool")
    preds = model.predict(pool_images, batch_size=BATCH_SIZE_PREDICT, verbose=0)
    
    # Calculate uncertainty (max probability entropy)
    uncertainty = 1 - np.max(preds, axis=1)
    
    # Select top K most uncertain samples
    top_indices = np.argsort(uncertainty)[-K:]
    x_uncertain = pool_images[top_indices]
    y_uncertain = pool_labels[top_indices]
    
    pipeline_log(f"Selected {len(x_uncertain)} uncertain samples for training")
    selected_count += len(x_uncertain)
    
    # Train on selected uncertain samples
    model.fit(x_uncertain, y_uncertain,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE_FIT,
              verbose=1)
    
    # Remove selected samples from pool
    pool_images = np.delete(pool_images, top_indices, axis=0)
    pool_labels = np.delete(pool_labels, top_indices, axis=0)
    
    # Evaluate on eval set
    preds_eval = model.predict(eval_images, batch_size=BATCH_SIZE_PREDICT, verbose=0)
    
    y_true = np.argmax(eval_labels, axis=1)
    y_pred = np.argmax(preds_eval, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    accuracy_list.append(acc)
    f1_list.append(f1)
    
    iter_elapsed = now_ms(iter_start)
    pipeline_log(f"AL Iteration {iteration} results | acc={acc:.4f} | f1={f1:.4f} | {iter_elapsed:.2f} ms")

# ----------------------------
# Save Active Learning Model
# ----------------------------
stage_start = start_stage("save_active_learning_model")
model.save("models/active_learning_subset_model.keras")
pipeline_log("Active Learning completed")
pipeline_log(f"Total samples used for training: {selected_count}")
pipeline_log(f"Remaining pool size: {len(pool_images)}")
pipeline_log("Model saved: models/active_learning_subset_model.keras")
end_stage("save_active_learning_model", stage_start)

# ----------------------------
# Save Active Learning Metrics
# ----------------------------
stage_start = start_stage("save_active_learning_metrics")
pipeline_log("Generating active learning performance graphs")

plt.figure(figsize=(10, 6))
plt.plot(range(1, ITERATIONS+1), accuracy_list, marker='o', linewidth=2, label='Accuracy', color='#2E86AB')
plt.plot(range(1, ITERATIONS+1), f1_list, marker='s', linewidth=2, label='Macro F1-score', color='#A23B72')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Active Learning Performance Over Iterations', fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("results/active_learning_accuracy.png", dpi=150)
plt.close()
pipeline_log("Saved: results/active_learning_accuracy.png")

# ----------------------------
# Save Active Learning Confusion Matrix
# ----------------------------
preds_final = model.predict(eval_images, batch_size=BATCH_SIZE_PREDICT, verbose=0)
y_true_final = np.argmax(eval_labels, axis=1)
y_pred_final = np.argmax(preds_final, axis=1)

cm = confusion_matrix(y_true_final, y_pred_final)

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap='Blues', aspect='auto')
plt.title("Confusion Matrix - Active Learning Final", fontsize=14, fontweight='bold')
plt.colorbar(label='Count')
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
class_names = list(train_gen.class_indices.keys())
plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
plt.yticks(range(num_classes), class_names)
plt.tight_layout()
plt.savefig("results/active_learning_confusion_matrix.png", dpi=150)
plt.close()
pipeline_log("Saved: results/active_learning_confusion_matrix.png")

pipeline_log("Active Learning Metrics")
pipeline_log(f"Final Accuracy: {accuracy_list[-1]:.4f}")
pipeline_log(f"Final F1-Score: {f1_list[-1]:.4f}")
pipeline_log(f"Accuracy Range: {min(accuracy_list):.4f} - {max(accuracy_list):.4f}")
end_stage("save_active_learning_metrics", stage_start)

# ============================================================
# CONTINUAL LEARNING PHASE
# ============================================================
print("\n" + "="*60)
print("===== CONTINUAL LEARNING PHASE =====")
print("="*60)
stage_start = start_stage("continual_learning_phase")

# Use remaining pool samples or up to CONTINUAL_SAMPLES
if len(pool_images) > 0:
    continual_samples = min(len(pool_images), CONTINUAL_SAMPLES)
    new_images = pool_images[:continual_samples]
    new_labels = pool_labels[:continual_samples]
    pipeline_log(f"Using {continual_samples} remaining samples for continual learning")
else:
    new_images = np.array([])
    new_labels = np.array([])
    print("⚠ No remaining pool samples for continual learning!")

if len(new_images) > 0:
    # Reduce learning rate for continual learning (very conservative to prevent degradation)
    model.compile(
        optimizer=Adam(learning_rate=CONTINUAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    pipeline_log(f"Training on {len(new_images)} samples with learning rate {CONTINUAL_LEARNING_RATE} (very conservative)")
    model.fit(new_images, new_labels,
              epochs=CONTINUAL_EPOCHS,
              batch_size=2,
              verbose=1)
    
    # Evaluate on test set
    preds_continual = model.predict(eval_images, batch_size=BATCH_SIZE_PREDICT, verbose=0)
    y_true_cont = np.argmax(eval_labels, axis=1)
    y_pred_cont = np.argmax(preds_continual, axis=1)
    
    continual_acc = accuracy_score(y_true_cont, y_pred_cont)
    continual_f1 = f1_score(y_true_cont, y_pred_cont, average='macro')
    
    pipeline_log("After Continual Learning")
    pipeline_log(f"Accuracy: {continual_acc:.4f}")
    pipeline_log(f"Macro F1: {continual_f1:.4f}")
    pipeline_log(f"Improvement: {(continual_acc - accuracy_list[-1])*100:+.2f}%")
    
    # ----------------------------
    # Save Continual Learning Confusion Matrix
    # ----------------------------
    cm_cont = confusion_matrix(y_true_cont, y_pred_cont)
    
    plt.figure(figsize=(8, 7))
    plt.imshow(cm_cont, cmap='Greens', aspect='auto')
    plt.title("Confusion Matrix - After Continual Learning", fontsize=14, fontweight='bold')
    plt.colorbar(label='Count')
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
    plt.yticks(range(num_classes), class_names)
    plt.tight_layout()
    plt.savefig("results/continual_learning_confusion_matrix.png", dpi=150)
    plt.close()
    pipeline_log("Saved: results/continual_learning_confusion_matrix.png")
    
    # Save final model
    model.save("models/continual_updated_model.h5")
    pipeline_log("Saved: models/continual_updated_model.h5")
else:
    print("⚠ Skipping continual learning (no samples available)")
end_stage("continual_learning_phase", stage_start)

# ----------------------------
# Visualize Sample Predictions
# ----------------------------
stage_start = start_stage("save_sample_predictions")
pipeline_log("Generating sample predictions visualization")
num_samples = min(5, len(eval_images))
plt.figure(figsize=(15, 3))

sample_preds = model.predict(eval_images[:num_samples], verbose=0)
for i in range(num_samples):
    plt.subplot(1, num_samples, i+1)
    img = eval_images[i]
    
    # Normalize for display if needed
    if img.max() <= 1:
        display_img = img
    else:
        display_img = img / 255.0
    
    plt.imshow(display_img)
    pred_class = np.argmax(sample_preds[i])
    confidence = np.max(sample_preds[i])
    true_class = np.argmax(eval_labels[i])
    
    title_color = 'green' if pred_class == true_class else 'red'
    plt.title(f"Pred: {class_names[pred_class]}\nConf: {confidence:.2f}", 
              color=title_color, fontsize=9, fontweight='bold')
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/sample_predictions.png", dpi=150)
plt.close()
pipeline_log("Saved: results/sample_predictions.png")
end_stage("save_sample_predictions", stage_start)

# ----------------------------
# Final Summary
# ----------------------------
print("\n" + "="*60)
print("✓ ALL PHASES COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Training Summary:")
print(f"  • Total training data: {len(images)} images (100% of available data)")
print(f"  • Active learning pool: {len(pool_images) + selected_count} images")
print(f"  • Samples trained in AL: {selected_count} images")
print(f"  • Active Learning iterations: {ITERATIONS}")
print(f"  • Final AL Accuracy: {accuracy_list[-1]:.4f}")
print(f"\n📁 Results saved to: results/")
print(f"📁 Models saved to: models/")

pipeline_total_ms = now_ms(pipeline_start)
pipeline_log("===== Pipeline Timing Summary (ms) =====")
for name, elapsed in stage_times_ms.items():
    pipeline_log(f"{name}: {elapsed:.2f} ms")
pipeline_log(f"TOTAL PIPELINE TIME: {pipeline_total_ms:.2f} ms")
pipeline_log("PIPELINE END")