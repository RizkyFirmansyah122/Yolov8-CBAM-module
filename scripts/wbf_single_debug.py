# scripts/wbf_single_debug.py
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import matplotlib.pyplot as plt

# --- CONFIG ---
IMAGE_PATH  = r"D:/TA/Project/cric/test/images/e501b86a4eb392736b083deb8c4537f8_png.rf.28b9a7b0c0fe18bfeea2229ad7df7a14.jpg"  # dataset yaml
MODEL_PATHS = [
    r"D:/TA/Project/ultralytics/models/v10l.pt",
    r"D:/TA/Project/ultralytics/models/v10m.pt",
    r"D:/TA/Project/ultralytics/models/v10s.pt",
]
DEVICE = 0  # CUDA
IMGSZ = 640
CONF_PER_MODEL = [0.15, 0.15, 0.15]
IOU_NMS_PER_MODEL = [0.6, 0.6, 0.6]
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001
WBF_WEIGHTS = [1.0, 1.0, 1.0]

# --- LOAD MODELS ---
models = [YOLO(p) for p in MODEL_PATHS]
im = cv2.imread(IMAGE_PATH)
h, w = im.shape[:2]

# --- COLORMAP per model untuk visual clarity ---
colors = [(255, 0, 0), (0, 255, 0), (0, 165, 255)]  # red, green, orange

# --- STEP 1: Run predictions from each model ---
all_boxes, all_scores, all_labels = [], [], []
for idx, (m, conf, iou) in enumerate(zip(models, CONF_PER_MODEL, IOU_NMS_PER_MODEL)):
    r = m.predict(source=IMAGE_PATH, imgsz=IMGSZ, conf=conf, iou=iou, device=DEVICE, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        continue
    b = r.boxes.xyxy.cpu().numpy()
    s = r.boxes.conf.cpu().numpy()
    c = r.boxes.cls.to(torch.int).cpu().numpy()

    # visualize individual model detections
    vis = im.copy()
    for box, score, cls in zip(b, s, c):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colors[idx], 2)
        cv2.putText(vis, f"{cls}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    cv2.imwrite(f"D:/TA/Project/ultralytics/outputwbf/model{idx+1}_pred.png", vis)

    # normalize coordinates for WBF
    b_norm = b.copy()
    b_norm[:, [0, 2]] /= w
    b_norm[:, [1, 3]] /= h
    all_boxes.append(b_norm.tolist())
    all_scores.append(s.tolist())
    all_labels.append(c.tolist())

# --- STEP 2: Apply Weighted Boxes Fusion ---
if sum(len(x) for x in all_boxes) == 0:
    print("No predictions to fuse.")
    exit()

boxes, scores, labels = weighted_boxes_fusion(
    all_boxes, all_scores, all_labels,
    weights=WBF_WEIGHTS,
    iou_thr=WBF_IOU_THR,
    skip_box_thr=WBF_SKIP_BOX_THR,
)

# --- Denormalize boxes to image coordinates ---
boxes[:, [0, 2]] *= w
boxes[:, [1, 3]] *= h

# --- STEP 3: Visualize ensemble result ---
ensemble_vis = im.copy()
for box, score, cls in zip(boxes, scores, labels):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(ensemble_vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(ensemble_vis, f"Fused {int(cls)}:{score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

save_path = Path(r"D:/TA/Project/ultralytics/outputwbf/wbf_fused_debug.png")
cv2.imwrite(str(save_path), ensemble_vis)
print(f"\n✅ Saved WBF ensemble visualization → {save_path}")

# Optional: show in matplotlib (inline visualization)
plt.imshow(cv2.cvtColor(ensemble_vis, cv2.COLOR_BGR2RGB))
plt.title("Weighted Boxes Fusion Result")
plt.axis("off")
plt.show()