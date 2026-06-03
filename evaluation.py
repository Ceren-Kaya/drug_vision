import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from load_preprocess import get_generators

MODEL_PATH = Path("outputs/best_model.keras")
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

train_images, val_images, test_images = get_generators()
model = tf.keras.models.load_model(MODEL_PATH)

idx_to_class = {v: k for k, v in train_images.class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# evaluate
test_loss, test_acc = model.evaluate(test_images, verbose=1)

print(f"\n[TEST] Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

(OUTPUT_DIR / "test_metrics.txt").write_text(
    f"Test Loss: {test_loss:.6f}\nTest Accuracy: {test_acc:.6f}\n",
    encoding="utf-8"
)

# prediction
test_images.reset()
y_prob = model.predict(test_images, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_images.classes 

top1_acc = np.mean(y_pred == y_true)

k = 3
top3_pred = np.argsort(y_prob, axis=1)[:, -k:]   # her satırın en büyük 3 index'i
top3_acc = np.mean([y_true[i] in top3_pred[i] for i in range(len(y_true))])

print(f"[TOP-1] Accuracy: {top1_acc:.4f}")
print(f"[TOP-3] Accuracy: {top3_acc:.4f}")

(OUTPUT_DIR / "topk_metrics.txt").write_text(
    f"Top-1 Accuracy: {top1_acc:.6f}\nTop-3 Accuracy: {top3_acc:.6f}\n",
    encoding="utf-8"
)

# classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report\n")
print(report)
(OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

# hard cases ( most confused class pairs)
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

flat_idx = np.argsort(cm_no_diag.ravel())[::-1]  # büyükten küçüğe
top_n = 5

lines = []
lines.append("Most Confused Pairs (True -> Pred):\n")
count = 0
for idx in flat_idx:
    i = idx // cm.shape[1]
    j = idx % cm.shape[1]
    val = cm_no_diag[i, j]
    if val == 0:
        break
    lines.append(f"{count+1}) {class_names[i]} -> {class_names[j]} : {val}\n")
    count += 1
    if count >= top_n:
        break

hard_cases_path = OUTPUT_DIR / "hard_cases.txt"
hard_cases_path.write_text("".join(lines), encoding="utf-8")
print(f"[SAVED] {hard_cases_path}")
