# scripts/wbf_val.py
from pathlib import Path
import yaml
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou, ap_per_class
from ensemble_boxes import weighted_boxes_fusion
import torch
import datetime as dt
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
DATA_YAML = r"D:/TA/Project/cric/data.yaml"  # dataset yaml
MODEL_PATHS = [
    r"D:/TA/Project/ultralytics/models/v10l.pt",
    r"D:/TA/Project/ultralytics/models/v10m.pt",
    r"D:/TA/Project/ultralytics/models/v10s.pt",
]

IMGSZ = 640
DEVICE = 0  # 0 for CUDA GPU, "cpu" for CPU
CONF_PER_MODEL = [0.4, 0.4, 0.4]
IOU_NMS_PER_MODEL = [0.50, 0.50, 0.50]

# WBF knobs
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001
WBF_WEIGHTS = [1.0, 1.0, 1.0]

# Evaluation IoUs
IOU_EVAL_THR = np.linspace(0.5, 0.95, 10)

# Confusion Matrix behavior
MATCH_IOU_FOR_CM = 0.50        # IoU untuk matching CM (greedy)
BUILD_WITH_BG = True           # bangun CM (nc+1)x(nc+1) lalu turunkan versi no-bg dari sini

OUTDIR = Path(r"D:/TA/Project/ultralytics/outputwbf")
OUTDIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_images_and_labels(cfg):
    base = Path(cfg.get("path", "."))
    val_img = cfg.get("val") or cfg.get("val_images")
    val_img = Path(val_img)
    if not val_img.is_absolute():
        val_img = base / val_img
    lbl = cfg.get("val_labels")
    if lbl:
        val_lbl = Path(lbl)
        if not val_lbl.is_absolute():
            val_lbl = base / val_lbl
    else:
        val_lbl = Path(str(val_img).replace("images", "labels"))
    return val_img, val_lbl


def list_images(folder):
    ims = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        ims.extend(list(Path(folder).rglob(ext)))
    return sorted(ims)


def read_labels_yolo(txt_path, w, h):
    if not txt_path.exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    arr = np.loadtxt(txt_path, ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    cls = arr[:, 0].astype(int)
    cx, cy, ww, hh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1 = (cx - ww / 2.0) * w
    y1 = (cy - hh / 2.0) * h
    x2 = (cx + ww / 2.0) * w
    y2 = (cy + hh / 2.0) * h
    return np.stack([x1, y1, x2, y2], axis=1), cls


def to_norm(xyxy, w, h):
    if xyxy.size == 0:
        return np.zeros((0, 4))
    x1, y1, x2, y2 = xyxy.T
    return np.stack([x1 / w, y1 / h, x2 / w, y2 / h], axis=1)


def from_norm(xyxy, w, h):
    if xyxy.size == 0:
        return np.zeros((0, 4))
    x1, y1, x2, y2 = xyxy.T
    return np.stack([x1 * w, y1 * h, x2 * w, y2 * h], axis=1)


def yolo_pred(model, img_path, conf, iou):
    """Run YOLO model inference on one image."""
    r = model.predict(source=img_path, imgsz=IMGSZ, conf=conf, iou=iou,
                      device=DEVICE, verbose=False, max_det=300)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    b = r.boxes.xyxy.cpu().numpy()
    s = r.boxes.conf.cpu().numpy()
    c = r.boxes.cls.to(torch.int).cpu().numpy()  # compat Ultralytics
    return b, s, c


def wbf_ensemble(img_path, models, confs, ious, wbf_iou, wbf_skip, weights):
    im = cv2.imread(str(img_path))
    h, w = im.shape[:2]
    all_b, all_s, all_c = [], [], []
    for m, cf, io in zip(models, confs, ious):
        b, s, c = yolo_pred(m, img_path, cf, io)
        all_b.append(to_norm(b, w, h).tolist())
        all_s.append(s.tolist())
        all_c.append(c.tolist())
    if sum(len(x) for x in all_b) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    B, S, L = weighted_boxes_fusion(all_b, all_s, all_c,
                                    weights=weights, iou_thr=wbf_iou, skip_box_thr=wbf_skip)
    return from_norm(np.array(B), w, h), np.array(S), np.array(L, dtype=int)


def plot_cm(cm, class_names, save_path, normalize=False, include_bg=True, title="Confusion Matrix"):
    """Plot confusion matrix. If normalize=True, values dinormalisasi per kolom (True class)."""
    M = cm.astype(np.float32).copy()
    if normalize:
        M = M / (M.sum(axis=0, keepdims=True) + 1e-9)

    plt.figure(figsize=(10, 7))
    im = plt.imshow(M, cmap="Blues", vmin=0.0)
    plt.colorbar(im, fraction=0.046, pad=0.05)

    ticks = class_names + (["background"] if include_bg else [])
    plt.xticks(np.arange(len(ticks)), ticks, rotation=90)
    plt.yticks(np.arange(len(ticks)), ticks)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title + (" (Normalized)" if normalize else ""))

    # tulis angka di setiap kotak
    H, W = M.shape
    # threshold untuk warna tulisan (biar kontras)
    thr = 0.45 * (1 if not normalize else np.nanmax(M))
    for i in range(H):
        for j in range(W):
            if normalize:
                if np.isnan(M[i, j]):
                    continue
                txt = f"{M[i, j]:.2f}"
                col = "white" if M[i, j] > thr else "black"
            else:
                txt = f"{int(cm[i, j])}"
                col = "white" if M[i, j] > thr else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def main():
    cfg = load_yaml(DATA_YAML)
    raw_names = cfg.get("names", {0: "0", 1: "1"})
    if isinstance(raw_names, list):
        names = {i: n for i, n in enumerate(raw_names)}
    else:
        names = {int(k): v for k, v in raw_names.items()}
    nc = len(names)
    class_names = [names[i] for i in range(nc)]

    models = [YOLO(p) for p in MODEL_PATHS]
    val_img_dir, val_lbl_dir = resolve_images_and_labels(cfg)
    images = list_images(val_img_dir)
    if len(images) == 0:
        print(f"No images found in {val_img_dir}")
        return

    tps, confs, pred_cls, target_cls = [], [], [], []

    # Bangun CM with-bg (nc+1 x nc+1). Nantinya kita turunkan CM no-bg dari sini.
    cm_bg = np.zeros((nc + 1, nc + 1), dtype=int) if BUILD_WITH_BG else np.zeros((nc, nc), dtype=int)

    for img_p in images:
        im = cv2.imread(str(img_p))
        h, w = im.shape[:2]
        gt_txt = Path(val_lbl_dir) / (Path(img_p).stem + ".txt")
        gt_xyxy, gt_cls = read_labels_yolo(gt_txt, w, h)

        p_xyxy, p_conf, p_cls = wbf_ensemble(
            img_p, models, CONF_PER_MODEL, IOU_NMS_PER_MODEL,
            WBF_IOU_THR, WBF_SKIP_BOX_THR, WBF_WEIGHTS
        )

        # ---------- AP/PR accumulation ----------
        iou = (
            box_iou(torch.tensor(gt_xyxy, dtype=torch.float32),
                    torch.tensor(p_xyxy, dtype=torch.float32)).numpy()
            if (len(p_xyxy) and len(gt_xyxy)) else np.zeros((len(gt_xyxy), len(p_xyxy)))
        )
        tp_img = np.zeros((p_xyxy.shape[0], IOU_EVAL_THR.size), dtype=bool)
        if iou.size:
            for j, thr in enumerate(IOU_EVAL_THR):
                gt_match = -np.ones(len(gt_xyxy), dtype=int)
                for pi in np.argsort(-p_conf):
                    gi = np.argmax(iou[:, pi]) if iou.shape[0] else -1
                    if gi >= 0 and iou[gi, pi] >= thr and gt_match[gi] == -1 and gt_cls[gi] == p_cls[pi]:
                        tp_img[pi, j] = True
                        gt_match[gi] = pi
        tps.append(tp_img)
        confs.append(p_conf)
        pred_cls.append(p_cls)
        target_cls.append(gt_cls)

        # ---------- Confusion Matrix accumulation (IoU=0.5 greedy) ----------
        iou_cm = (
            box_iou(torch.tensor(gt_xyxy, dtype=torch.float32),
                    torch.tensor(p_xyxy, dtype=torch.float32)).numpy()
            if (len(p_xyxy) and len(gt_xyxy)) else np.zeros((len(gt_xyxy), len(p_xyxy)))
        )
        gt_match_cm = -np.ones(len(gt_xyxy), dtype=int)
        used_pred = set()
        for pi in np.argsort(-p_conf):  # greedy dari conf tertinggi
            gi = np.argmax(iou_cm[:, pi]) if iou_cm.shape[0] else -1
            if gi >= 0 and iou_cm[gi, pi] >= MATCH_IOU_FOR_CM and gt_match_cm[gi] == -1:
                # predicted vs true
                cm_bg[p_cls[pi], gt_cls[gi]] += 1
                gt_match_cm[gi] = pi
                used_pred.add(pi)

        if BUILD_WITH_BG:
            # pred tak terpakai -> True=bg (kolom terakhir)
            for pi in range(len(p_cls)):
                if pi not in used_pred:
                    cm_bg[p_cls[pi], nc] += 1
            # GT tak ter-match -> Pred=bg (baris terakhir)
            for gi in range(len(gt_cls)):
                if gt_match_cm[gi] == -1:
                    cm_bg[nc, gt_cls[gi]] += 1

    if len(confs) == 0:
        print("No statistics collected; check your paths/labels.")
        return

    tps = np.concatenate(tps, 0)
    confs = np.concatenate(confs, 0)
    pred_cls = np.concatenate(pred_cls, 0)
    target_cls = np.concatenate(target_cls, 0)

    # ap_per_class -> (tp, fp, p, r, f1, ap, unique, p_curve, r_curve, f1_curve, x, prec_values)
    _, _, P, R, F1, AP, unique, p_curve, r_curve, f1_curve, x_vals, prec_values = ap_per_class(
        tps, confs, pred_cls, target_cls, plot=False, save_dir=OUTDIR
    )

    mean_P = float(np.nanmean(P)) if P.size else 0.0
    mean_R = float(np.nanmean(R)) if R.size else 0.0
    mAP50 = float(np.nanmean(AP[:, 0])) if AP.size else 0.0
    mAP50_95 = float(np.nanmean(AP)) if AP.size else 0.0
    mean_F1 = float(np.nanmean(F1)) if F1.size else 0.0

    print("\n=== WBF Evaluation (headline) ===")
    print(f"Precision (mean):  {mean_P:.4f}")
    print(f"Recall    (mean):  {mean_R:.4f}")
    print(f"F1        (mean):  {mean_F1:.4f}")
    print(f"mAP@0.5          : {mAP50:.4f}")
    print(f"mAP@0.5:0.95     : {mAP50_95:.4f}")

    # --------- Save Confusion Matrices (4 variasi) ----------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # # 1) WITH BG: raw counts
    # path_counts_bg = OUTDIR / f"cm_counts_with_bg_{ts}.png"
    # plot_cm(cm_bg, class_names, path_counts_bg, normalize=False, include_bg=True,
    #         title="Confusion Matrix (counts, with background)")
    # # 2) WITH BG: normalized
    # path_norm_bg = OUTDIR / f"cm_norm_with_bg_{ts}.png"
    # plot_cm(cm_bg, class_names, path_norm_bg, normalize=True, include_bg=True,
    #         title="Confusion Matrix (normalized, with background)")

    # 3) NO BG: raw counts (crop)
    cm_no_bg = cm_bg[:nc, :nc] if BUILD_WITH_BG else cm_bg  # aman bila BUILD_WITH_BG=False
    path_counts_nobg = OUTDIR / f"cm_counts_no_bg_{ts}.png"
    plot_cm(cm_no_bg, class_names, path_counts_nobg, normalize=False, include_bg=False,
            title="Confusion Matrix (counts, no background)")
    # 4) NO BG: normalized
    path_norm_nobg = OUTDIR / f"cm_norm_no_bg_{ts}.png"
    plot_cm(cm_no_bg, class_names, path_norm_nobg, normalize=True, include_bg=False,
            title="Confusion Matrix (normalized, no background)")

    print("\nSaved confusion matrices:")
    # print(f"- {path_counts_bg}")
    # print(f"- {path_norm_bg}")
    print(f"- {path_counts_nobg}")
    print(f"- {path_norm_nobg}")


if __name__ == "__main__":
    main()





# # CODE KEDUA

# # scripts/wbf_val.py
# from pathlib import Path
# import yaml
# import numpy as np
# import cv2
# from ultralytics import YOLO
# from ultralytics.utils.metrics import box_iou, ap_per_class
# from ensemble_boxes import weighted_boxes_fusion
# import torch
# import datetime as dt
# import matplotlib.pyplot as plt

# # -------------------- CONFIG --------------------
# DATA_YAML = r"D:/TA/Project/cric/data.yaml"  # dataset yaml
# MODEL_PATHS = [
#     r"D:/TA/Project/ultralytics/models/v10l.pt",
#     r"D:/TA/Project/ultralytics/models/v10m.pt",
#     r"D:/TA/Project/ultralytics/models/v10s.pt",
# ]

# IMGSZ = 640
# DEVICE = 0  # 0 for CUDA GPU, "cpu" for CPU
# CONF_PER_MODEL = [0.4, 0.4, 0.4]
# IOU_NMS_PER_MODEL = [0.50, 0.50, 0.50]

# # WBF knobs
# WBF_IOU_THR = 0.55
# WBF_SKIP_BOX_THR = 0.001
# WBF_WEIGHTS = [1.0, 1.0, 1.0]

# # Evaluation IoUs
# IOU_EVAL_THR = np.linspace(0.5, 0.95, 10)

# # Confusion Matrix behavior
# MATCH_IOU_FOR_CM = 0.50        # IoU untuk matching CM
# INCLUDE_BG_IN_CM = True        # True => (nc+1)x(nc+1) dengan 'bg'; False => nc x nc tanpa 'bg'

# OUTDIR = Path(r"D:/TA/Project/ultralytics/outputwbf")
# OUTDIR.mkdir(parents=True, exist_ok=True)
# # ------------------------------------------------


# def load_yaml(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def resolve_images_and_labels(cfg):
#     base = Path(cfg.get("path", "."))
#     val_img = cfg.get("val") or cfg.get("val_images")
#     val_img = Path(val_img)
#     if not val_img.is_absolute():
#         val_img = base / val_img
#     # guess labels dir
#     lbl = cfg.get("val_labels")
#     if lbl:
#         val_lbl = Path(lbl)
#         if not val_lbl.is_absolute():
#             val_lbl = base / val_lbl
#     else:
#         val_lbl = Path(str(val_img).replace("images", "labels"))
#     return val_img, val_lbl


# def list_images(folder):
#     ims = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
#         ims.extend(list(Path(folder).rglob(ext)))
#     return sorted(ims)


# def read_labels_yolo(txt_path, w, h):
#     if not txt_path.exists():
#         return np.zeros((0, 4)), np.zeros((0,), dtype=int)
#     arr = np.loadtxt(txt_path, ndmin=2)
#     if arr.size == 0:
#         return np.zeros((0, 4)), np.zeros((0,), dtype=int)
#     cls = arr[:, 0].astype(int)
#     cx, cy, ww, hh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
#     x1 = (cx - ww / 2.0) * w
#     y1 = (cy - hh / 2.0) * h
#     x2 = (cx + ww / 2.0) * w
#     y2 = (cy + hh / 2.0) * h
#     return np.stack([x1, y1, x2, y2], axis=1), cls


# def to_norm(xyxy, w, h):
#     if xyxy.size == 0:
#         return np.zeros((0, 4))
#     x1, y1, x2, y2 = xyxy.T
#     return np.stack([x1 / w, y1 / h, x2 / w, y2 / h], axis=1)


# def from_norm(xyxy, w, h):
#     if xyxy.size == 0:
#         return np.zeros((0, 4))
#     x1, y1, x2, y2 = xyxy.T
#     return np.stack([x1 * w, y1 * h, x2 * w, y2 * h], axis=1)


# def yolo_pred(model, img_path, conf, iou):
#     """
#     Run YOLO model inference on one image.
#     """
#     r = model.predict(source=img_path, imgsz=IMGSZ, conf=conf, iou=iou,
#                       device=DEVICE, verbose=False, max_det=300)[0]
#     if r.boxes is None or len(r.boxes) == 0:
#         return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
#     b = r.boxes.xyxy.cpu().numpy()
#     s = r.boxes.conf.cpu().numpy()
#     c = r.boxes.cls.to(torch.int).cpu().numpy()  # compat
#     return b, s, c


# def wbf_ensemble(img_path, models, confs, ious, wbf_iou, wbf_skip, weights):
#     im = cv2.imread(str(img_path))
#     h, w = im.shape[:2]
#     all_b, all_s, all_c = [], [], []
#     for m, cf, io in zip(models, confs, ious):
#         b, s, c = yolo_pred(m, img_path, cf, io)
#         all_b.append(to_norm(b, w, h).tolist())
#         all_s.append(s.tolist())
#         all_c.append(c.tolist())
#     if sum(len(x) for x in all_b) == 0:
#         return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
#     B, S, L = weighted_boxes_fusion(all_b, all_s, all_c,
#                                     weights=weights, iou_thr=wbf_iou, skip_box_thr=wbf_skip)
#     return from_norm(np.array(B), w, h), np.array(S), np.array(L, dtype=int)


# def plot_confusion_matrix(cm, class_names, save_path, normalize=True, include_bg=True):
#     cm_plot = cm.astype(np.float32)
#     if normalize:
#         # normalize per-column (True class)
#         cm_plot = cm_plot / (cm_plot.sum(axis=0, keepdims=True) + 1e-9)

#     plt.figure(figsize=(8, 6))
#     im = plt.imshow(cm_plot, cmap="Blues", vmin=0.0)
#     plt.colorbar(im, fraction=0.046, pad=0.05)

#     ticks = class_names + (["bg"] if include_bg else [])
#     plt.xticks(np.arange(len(ticks)), ticks, rotation=90)
#     plt.yticks(np.arange(len(ticks)), ticks)
#     plt.xlabel("True")
#     plt.ylabel("Predicted")
#     plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

#     # annotate values for small matrices
#     if cm_plot.shape[0] <= 20:
#         thresh = 0.45 * (1 if not normalize else np.nanmax(cm_plot))
#         for i in range(cm_plot.shape[0]):
#             for j in range(cm_plot.shape[1]):
#                 v = cm_plot[i, j]
#                 if np.isnan(v):
#                     continue
#                 txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
#                 plt.text(j, i, txt, ha="center", va="center",
#                          color="white" if (normalize and v > thresh) or (not normalize and cm_plot[i, j] > thresh) else "black",
#                          fontsize=8)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=250)
#     plt.close()


# def main():
#     cfg = load_yaml(DATA_YAML)
#     raw_names = cfg.get("names", {0: "0", 1: "1"})
#     if isinstance(raw_names, list):
#         names = {i: n for i, n in enumerate(raw_names)}
#     else:
#         names = {int(k): v for k, v in raw_names.items()}
#     nc = len(names)

#     models = [YOLO(p) for p in MODEL_PATHS]
#     val_img_dir, val_lbl_dir = resolve_images_and_labels(cfg)
#     images = list_images(val_img_dir)
#     if len(images) == 0:
#         print(f"No images found in {val_img_dir}")
#         return

#     tps, confs, pred_cls, target_cls = [], [], [], []

#     # Confusion matrix init
#     if INCLUDE_BG_IN_CM:
#         cm = np.zeros((nc + 1, nc + 1), dtype=int)  # last index = bg
#     else:
#         cm = np.zeros((nc, nc), dtype=int)          # strictly nc x nc

#     for img_p in images:
#         im = cv2.imread(str(img_p))
#         h, w = im.shape[:2]
#         gt_txt = Path(val_lbl_dir) / (Path(img_p).stem + ".txt")
#         gt_xyxy, gt_cls = read_labels_yolo(gt_txt, w, h)

#         p_xyxy, p_conf, p_cls = wbf_ensemble(
#             img_p, models, CONF_PER_MODEL, IOU_NMS_PER_MODEL,
#             WBF_IOU_THR, WBF_SKIP_BOX_THR, WBF_WEIGHTS
#         )

#         # ---------- AP/PR accumulation ----------
#         iou = (
#             box_iou(torch.tensor(gt_xyxy, dtype=torch.float32),
#                     torch.tensor(p_xyxy, dtype=torch.float32)).numpy()
#             if (len(p_xyxy) and len(gt_xyxy)) else np.zeros((len(gt_xyxy), len(p_xyxy)))
#         )
#         tp_img = np.zeros((p_xyxy.shape[0], IOU_EVAL_THR.size), dtype=bool)
#         if iou.size:
#             for j, thr in enumerate(IOU_EVAL_THR):
#                 gt_match = -np.ones(len(gt_xyxy), dtype=int)
#                 for pi in np.argsort(-p_conf):
#                     gi = np.argmax(iou[:, pi]) if iou.shape[0] else -1
#                     if gi >= 0 and iou[gi, pi] >= thr and gt_match[gi] == -1 and gt_cls[gi] == p_cls[pi]:
#                         tp_img[pi, j] = True
#                         gt_match[gi] = pi
#         tps.append(tp_img)
#         confs.append(p_conf)
#         pred_cls.append(p_cls)
#         target_cls.append(gt_cls)

#         # ---------- Confusion Matrix accumulation (IoU=0.5 greedy) ----------
#         # Kita bangun CM dengan greedy matching sekali (threshold MATCH_IOU_FOR_CM)
#         if len(p_xyxy) or len(gt_xyxy):
#             iou_cm = (
#                 box_iou(torch.tensor(gt_xyxy, dtype=torch.float32),
#                         torch.tensor(p_xyxy, dtype=torch.float32)).numpy()
#                 if (len(p_xyxy) and len(gt_xyxy)) else np.zeros((len(gt_xyxy), len(p_xyxy)))
#             )
#             gt_match_cm = -np.ones(len(gt_xyxy), dtype=int)
#             used_pred = set()
#             # match berdasarkan confidence tinggi ke rendah
#             for pi in np.argsort(-p_conf):
#                 gi = np.argmax(iou_cm[:, pi]) if iou_cm.shape[0] else -1
#                 if gi >= 0 and iou_cm[gi, pi] >= MATCH_IOU_FOR_CM and gt_match_cm[gi] == -1:
#                     # matched: predicted label vs true label
#                     cm[p_cls[pi], gt_cls[gi]] += 1
#                     gt_match_cm[gi] = pi
#                     used_pred.add(pi)

#             if INCLUDE_BG_IN_CM:
#                 # unmatched predictions -> Pred=cls vs True=bg (last col)
#                 for pi in range(len(p_cls)):
#                     if pi not in used_pred:
#                         cm[p_cls[pi], nc] += 1
#                 # unmatched GT -> Pred=bg (last row) vs True=cls
#                 for gi in range(len(gt_cls)):
#                     if gt_match_cm[gi] == -1:
#                         cm[nc, gt_cls[gi]] += 1
#             # else (tanpa BG) kita abaikan unmatched untuk menjaga matriks nc x nc

#     if len(confs) == 0:
#         print("No statistics collected; check your paths/labels.")
#         return

#     tps = np.concatenate(tps, 0)
#     confs = np.concatenate(confs, 0)
#     pred_cls = np.concatenate(pred_cls, 0)
#     target_cls = np.concatenate(target_cls, 0)

#     # ap_per_class -> (tp, fp, p, r, f1, ap, unique, p_curve, r_curve, f1_curve, x, prec_values)
#     _, _, P, R, F1, AP, unique, p_curve, r_curve, f1_curve, x_vals, prec_values = ap_per_class(
#         tps, confs, pred_cls, target_cls, plot=False, save_dir=OUTDIR
#     )

#     mean_P = float(np.nanmean(P)) if P.size else 0.0
#     mean_R = float(np.nanmean(R)) if R.size else 0.0
#     mAP50 = float(np.nanmean(AP[:, 0])) if AP.size else 0.0
#     mAP50_95 = float(np.nanmean(AP)) if AP.size else 0.0
#     mean_F1 = float(np.nanmean(F1)) if F1.size else 0.0

#     print("\n=== WBF Evaluation (headline) ===")
#     print(f"Precision (mean):  {mean_P:.4f}")
#     print(f"Recall    (mean):  {mean_R:.4f}")
#     print(f"F1        (mean):  {mean_F1:.4f}")
#     print(f"mAP@0.5          : {mAP50:.4f}")
#     print(f"mAP@0.5:0.95     : {mAP50_95:.4f}")

#     # --------- Confusion Matrix Heatmap (save to file) ----------
#     ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
#     cm_path = OUTDIR / f"confusion_matrix_{'with_bg' if INCLUDE_BG_IN_CM else 'no_bg'}_{ts}.png"
#     class_names = [names[i] for i in range(nc)]
#     plot_confusion_matrix(cm, class_names, cm_path, normalize=True, include_bg=INCLUDE_BG_IN_CM)
#     print(f"Confusion matrix saved → {cm_path}")


# if __name__ == "__main__":
#     main()

# CODE PERTAMA

# # scripts/wbf_val.py
# from pathlib import Path
# import yaml
# import numpy as np
# import cv2
# from ultralytics import YOLO
# from ultralytics.utils.metrics import box_iou, ap_per_class
# from ensemble_boxes import weighted_boxes_fusion
# import torch
# import csv
# import json
# import datetime as dt
# import matplotlib.pyplot as plt

# # -------------------- CONFIG --------------------
# DATA_YAML = r"D:/TA/Project/cric/data.yaml"  # dataset yaml
# MODEL_PATHS = [
#     r"D:/TA/Project/ultralytics/models/v10l.pt",
#     r"D:/TA/Project/ultralytics/models/v10m.pt",
#     r"D:/TA/Project/ultralytics/models/v10s.pt",
# ]

# IMGSZ = 640
# DEVICE = 0  # 0 for CUDA GPU, "cpu" for CPU
# CONF_PER_MODEL = [0.4, 0.4, 0.4]
# IOU_NMS_PER_MODEL = [0.50, 0.50, 0.50]

# # WBF knobs
# WBF_IOU_THR = 0.55
# WBF_SKIP_BOX_THR = 0.001
# WBF_WEIGHTS = [1.0, 1.0, 1.0]

# # Evaluation IoUs
# IOU_EVAL_THR = np.linspace(0.5, 0.95, 10)
# MATCH_IOU_FOR_CM = 0.50  # IoU untuk confusion-matrix matching

# OUTDIR = Path(r"D:/TA/Project/ultralytics/outputwbf")
# OUTDIR.mkdir(parents=True, exist_ok=True)
# # ------------------------------------------------


# def load_yaml(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def resolve_images_and_labels(cfg):
#     base = Path(cfg.get("path", "."))
#     val_img = cfg.get("val") or cfg.get("val_images")
#     val_img = Path(val_img)
#     if not val_img.is_absolute():
#         val_img = base / val_img
#     # guess labels dir
#     lbl = cfg.get("val_labels")
#     if lbl:
#         val_lbl = Path(lbl)
#         if not val_lbl.is_absolute():
#             val_lbl = base / val_lbl
#     else:
#         val_lbl = Path(str(val_img).replace("images", "labels"))
#     return val_img, val_lbl


# def list_images(folder):
#     ims = []
#     for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
#         ims.extend(list(Path(folder).rglob(ext)))
#     return sorted(ims)


# def read_labels_yolo(txt_path, w, h):
#     if not txt_path.exists():
#         return np.zeros((0, 4)), np.zeros((0,), dtype=int)
#     arr = np.loadtxt(txt_path, ndmin=2)
#     if arr.size == 0:
#         return np.zeros((0, 4)), np.zeros((0,), dtype=int)
#     cls = arr[:, 0].astype(int)
#     cx, cy, ww, hh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
#     x1 = (cx - ww / 2.0) * w
#     y1 = (cy - hh / 2.0) * h
#     x2 = (cx + ww / 2.0) * w
#     y2 = (cy + hh / 2.0) * h
#     return np.stack([x1, y1, x2, y2], axis=1), cls


# def to_norm(xyxy, w, h):
#     if xyxy.size == 0:
#         return np.zeros((0, 4))
#     x1, y1, x2, y2 = xyxy.T
#     return np.stack([x1 / w, y1 / h, x2 / w, y2 / h], axis=1)


# def from_norm(xyxy, w, h):
#     if xyxy.size == 0:
#         return np.zeros((0, 4))
#     x1, y1, x2, y2 = xyxy.T
#     return np.stack([x1 * w, y1 * h, x2 * w, y2 * h], axis=1)


# def yolo_pred(model, img_path, conf, iou):
#     """
#     Run YOLO model inference on one image.
#     """
#     r = model.predict(source=img_path, imgsz=IMGSZ, conf=conf, iou=iou,
#                       device=DEVICE, verbose=False, max_det=300)[0]
#     if r.boxes is None or len(r.boxes) == 0:
#         return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
#     b = r.boxes.xyxy.cpu().numpy()
#     s = r.boxes.conf.cpu().numpy()
#     c = r.boxes.cls.to(torch.int).cpu().numpy()  # ✅ FIXED (compat with new Ultralytics)
#     return b, s, c


# def wbf_ensemble(img_path, models, confs, ious, wbf_iou, wbf_skip, weights):
#     im = cv2.imread(str(img_path))
#     h, w = im.shape[:2]
#     all_b, all_s, all_c = [], [], []
#     for m, cf, io in zip(models, confs, ious):
#         b, s, c = yolo_pred(m, img_path, cf, io)
#         all_b.append(to_norm(b, w, h).tolist())
#         all_s.append(s.tolist())
#         all_c.append(c.tolist())
#     if sum(len(x) for x in all_b) == 0:
#         return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
#     B, S, L = weighted_boxes_fusion(all_b, all_s, all_c,
#                                     weights=weights, iou_thr=wbf_iou, skip_box_thr=wbf_skip)
#     return from_norm(np.array(B), w, h), np.array(S), np.array(L, dtype=int)


# def plot_pr_curve(px, prec_values, AP, names, save_path):
#     plt.figure(figsize=(9, 6))
#     if 0 < len(names) < 21 and prec_values.shape[0] == len(names):
#         for i in range(prec_values.shape[0]):
#             label = f"{names[i]} {AP[i, 0]:.3f}"
#             plt.plot(px, prec_values[i], linewidth=1, label=label)
#         plt.legend(loc="upper right")
#     else:
#         plt.plot(px, prec_values.mean(0), linewidth=2, label="mean")
#         plt.legend(loc="upper right")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision–Recall Curve")
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.grid(True, alpha=0.25)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=250)
#     plt.close()


# def plot_mc_curve(px, f1_curve, save_path, ylabel="F1"):
#     plt.figure(figsize=(9, 6))
#     plt.plot(px, f1_curve.mean(0), linewidth=2, label="mean")
#     plt.xlabel("Confidence")
#     plt.ylabel(ylabel)
#     plt.title(f"{ylabel}–Confidence Curve")
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.grid(True, alpha=0.25)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=250)
#     plt.close()


# def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
#     cm_plot = cm.astype(np.float32)
#     if normalize:
#         cm_plot = cm_plot / (cm_plot.sum(axis=0, keepdims=True) + 1e-9)
#     plt.figure(figsize=(8, 6))
#     im = plt.imshow(cm_plot, cmap="Blues", vmin=0.0)
#     plt.colorbar(im, fraction=0.046, pad=0.05)
#     ticklabels = class_names + ["bg"]
#     plt.xticks(np.arange(len(ticklabels)), ticklabels, rotation=90)
#     plt.yticks(np.arange(len(ticklabels)), ticklabels)
#     plt.xlabel("True")
#     plt.ylabel("Predicted")
#     plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
#     if cm_plot.shape[0] <= 15:
#         thresh = 0.45 * (1 if not normalize else np.nanmax(cm_plot))
#         for i in range(cm_plot.shape[0]):
#             for j in range(cm_plot.shape[1]):
#                 v = cm_plot[i, j]
#                 if np.isnan(v):
#                     continue
#                 txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
#                 plt.text(j, i, txt, ha="center", va="center",
#                          color="white" if (normalize and v > thresh) or (not normalize and cm_plot[i, j] > thresh) else "black",
#                          fontsize=8)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=250)
#     plt.close()


# def main():
#     cfg = load_yaml(DATA_YAML)
#     raw_names = cfg.get("names", {0: "0", 1: "1"})
#     if isinstance(raw_names, list):
#         names = {i: n for i, n in enumerate(raw_names)}
#     else:
#         names = {int(k): v for k, v in raw_names.items()}
#     nc = len(names)

#     models = [YOLO(p) for p in MODEL_PATHS]
#     val_img_dir, val_lbl_dir = resolve_images_and_labels(cfg)
#     images = list_images(val_img_dir)
#     if len(images) == 0:
#         print(f"No images found in {val_img_dir}")
#         return

#     tps, confs, pred_cls, target_cls = [], [], [], []
#     cm = np.zeros((nc + 1, nc + 1), dtype=int)

#     for img_p in images:
#         im = cv2.imread(str(img_p))
#         h, w = im.shape[:2]
#         gt_txt = Path(val_lbl_dir) / (Path(img_p).stem + ".txt")
#         gt_xyxy, gt_cls = read_labels_yolo(gt_txt, w, h)

#         p_xyxy, p_conf, p_cls = wbf_ensemble(
#             img_p, models, CONF_PER_MODEL, IOU_NMS_PER_MODEL,
#             WBF_IOU_THR, WBF_SKIP_BOX_THR, WBF_WEIGHTS
#         )

#         # AP/PR accumulation
#         iou = (
#             box_iou(torch.tensor(gt_xyxy, dtype=torch.float32),
#                     torch.tensor(p_xyxy, dtype=torch.float32)).numpy()
#             if (len(p_xyxy) and len(gt_xyxy)) else np.zeros((len(gt_xyxy), len(p_xyxy)))
#         )
#         tp_img = np.zeros((p_xyxy.shape[0], IOU_EVAL_THR.size), dtype=bool)
#         if iou.size:
#             for j, thr in enumerate(IOU_EVAL_THR):
#                 gt_match = -np.ones(len(gt_xyxy), dtype=int)
#                 for pi in np.argsort(-p_conf):
#                     gi = np.argmax(iou[:, pi]) if iou.shape[0] else -1
#                     if gi >= 0 and iou[gi, pi] >= thr and gt_match[gi] == -1 and gt_cls[gi] == p_cls[pi]:
#                         tp_img[pi, j] = True
#                         gt_match[gi] = pi
#         tps.append(tp_img)
#         confs.append(p_conf)
#         pred_cls.append(p_cls)
#         target_cls.append(gt_cls)

#     if len(confs) == 0:
#         print("No statistics collected; check your paths/labels.")
#         return

#     tps = np.concatenate(tps, 0)
#     confs = np.concatenate(confs, 0)
#     pred_cls = np.concatenate(pred_cls, 0)
#     target_cls = np.concatenate(target_cls, 0)

#     _, _, P, R, F1, AP, unique, p_curve, r_curve, f1_curve, x_vals, prec_values = ap_per_class(
#         tps, confs, pred_cls, target_cls, plot=False, save_dir=OUTDIR
#     )

#     mean_P = float(np.nanmean(P)) if P.size else 0.0
#     mean_R = float(np.nanmean(R)) if R.size else 0.0
#     mAP50 = float(np.nanmean(AP[:, 0])) if AP.size else 0.0
#     mAP50_95 = float(np.nanmean(AP)) if AP.size else 0.0
#     mean_F1 = float(np.nanmean(F1)) if F1.size else 0.0

#     print("\n=== WBF Evaluation (headline) ===")
#     print(f"Precision (mean):  {mean_P:.4f}")
#     print(f"Recall    (mean):  {mean_R:.4f}")
#     print(f"F1        (mean):  {mean_F1:.4f}")
#     print(f"mAP@0.5          : {mAP50:.4f}")
#     print(f"mAP@0.5:0.95     : {mAP50_95:.4f}")


# if __name__ == "__main__":
#     main()
