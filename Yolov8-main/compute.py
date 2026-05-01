from ultralytics import YOLO
import os
import time
import psutil
import torch
import numpy as np
from pathlib import Path



def get_memory_usage():
    """Return RSS memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_cpu_usage():
    """Return process CPU usage percentage."""
    return psutil.Process(os.getpid()).cpu_percent(interval=None)


#
#  Accuracy metric helpers
#

def box_iou_numpy(boxes1, boxes2):
    """
    Compute pairwise IoU for boxes in [N, 4] format (x1, y1, x2, y2).
    Returns an [N, M] IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def load_gt_labels(label_path, img_shape):
    """
    Load YOLO ground-truth labels as np.ndarray [N, 5] (cls, x1, y1, x2, y2).
    """
    h, w = img_shape[:2]
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                labels.append([cls, x1, y1, x2, y2])
    return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)


def compute_metrics_per_image(pred_boxes, pred_cls, pred_conf, gt, iou_thres=0.5):
    """
    Compute TP, FP, and FN values for one image.
    pred_boxes : np.ndarray [M, 4] (x1, y1, x2, y2)
    pred_cls   : np.ndarray [M] int
    pred_conf  : np.ndarray [M] float
    gt         : np.ndarray [N, 5] (cls, x1, y1, x2, y2)
    Returns: {cls_id: {'tp', 'fp', 'fn', 'conf_list'}}
    """
    result = {}
    gt_boxes = gt[:, 1:]  # [N, 4]
    gt_cls = gt[:, 0].astype(int)
    gt_matched = np.zeros(len(gt), dtype=bool)

    for m in range(len(pred_boxes)):
        pc = int(pred_cls[m])
        conf = float(pred_conf[m])
        if pc not in result:
            result[pc] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}

        same_mask = (gt_cls == pc)
        if same_mask.sum() == 0:
            result[pc]['fp'] += 1
            result[pc]['conf_list'].append((conf, 0))
            continue

        ious = box_iou_numpy(pred_boxes[m:m + 1], gt_boxes[same_mask])[0]  # [K]
        max_iou_idx = ious.argmax()
        max_iou = ious[max_iou_idx]
        global_indices = np.where(same_mask)[0]
        global_idx = global_indices[max_iou_idx]

        if max_iou >= iou_thres and not gt_matched[global_idx]:
            result[pc]['tp'] += 1
            result[pc]['conf_list'].append((conf, 1))
            gt_matched[global_idx] = True
        else:
            result[pc]['fp'] += 1
            result[pc]['conf_list'].append((conf, 0))

    # Count unmatched ground-truth boxes as false negatives.
    for n in range(len(gt)):
        if not gt_matched[n]:
            gc = int(gt_cls[n])
            if gc not in result:
                result[gc] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}
            result[gc]['fn'] += 1

    return result


def merge_metrics(global_metrics, image_metrics):
    for cls_id, vals in image_metrics.items():
        if cls_id not in global_metrics:
            global_metrics[cls_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}
        global_metrics[cls_id]['tp'] += vals['tp']
        global_metrics[cls_id]['fp'] += vals['fp']
        global_metrics[cls_id]['fn'] += vals['fn']
        global_metrics[cls_id]['conf_list'] += vals['conf_list']


def compute_ap(conf_list, total_gt):
    """Compute 11-point interpolation AP."""
    if total_gt == 0 or not conf_list:
        return 0.0
    conf_list = sorted(conf_list, key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for _, is_tp in conf_list:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(prec) if prec else 0.0
    return ap / 11


def print_metrics_table(global_metrics, names, iou_thres):
    print(f"\n{'=' * 74}")
    print(f"  Accuracy evaluation results  (IoU threshold = {iou_thres})")
    print(f"{'=' * 74}")
    print(f"  {'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP':>8}")
    print(f"  {'-' * 72}")

    all_tp = all_fp = all_fn = 0
    all_ap = []

    for cls_id, vals in sorted(global_metrics.items()):
        tp, fp, fn = vals['tp'], vals['fp'], vals['fn']
        total_gt = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        ap = compute_ap(vals['conf_list'], total_gt)
        # Ultralytics model.names may be a dict {int: str}.
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (
            names[cls_id] if cls_id < len(names) else str(cls_id))
        print(f"  {cls_name:<20} {tp:>6} {fp:>6} {fn:>6} {prec:>8.3f} {recall:>8.3f} {f1:>8.3f} {ap:>8.3f}")
        all_tp += tp;
        all_fp += fp;
        all_fn += fn
        all_ap.append(ap)

    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = 2 * all_prec * all_recall / (all_prec + all_recall) if (all_prec + all_recall) > 0 else 0.0
    mAP = sum(all_ap) / len(all_ap) if all_ap else 0.0
    print(f"  {'-' * 72}")
    print(
        f"  {'ALL':<20} {all_tp:>6} {all_fp:>6} {all_fn:>6} {all_prec:>8.3f} {all_recall:>8.3f} {all_f1:>8.3f} {mAP:>8.3f}   mAP")
    print(f"{'=' * 74}\n")


#  Main workflow

IOU_THRES = 0.5  # IoU threshold

weight_path = r"./runs/detect/train320/weights/best.pt"
image_folder = "datasets/animal_yolo_format/images/test"

model = YOLO(weight_path)
names = model.names  # {0: 'cat', 1: 'dog', ...}

image_paths = [
    os.path.join(image_folder, img)
    for img in os.listdir(image_folder)
    if img.endswith(('.jpg', '.png', '.jpeg'))
]

# Global statistics containers.
all_perf_stats = []
global_metrics = {}
get_cpu_usage()  # warm up cpu_percent
t0 = time.time()

for image_path in image_paths:

    # Start performance sampling.
    mem_before = get_memory_usage()
    t_img_start = time.perf_counter()

    results = model.predict(
        task="detect",
        source=image_path,
        imgsz=320,
        max_det=1000,
        conf=0.60,
        iou=IOU_THRES,
        show_labels=False,
        show_conf=True,
        save=True,
        device="cpu",
        augment=True,
        verbose=False,
    )

    # Finish performance sampling.
    t_img_end = time.perf_counter()
    mem_after = get_memory_usage()
    cpu_pct = get_cpu_usage()
    total_ms = (t_img_end - t_img_start) * 1e3
    mem_delta = mem_after - mem_before
    for result in results:
        img_shape = result.orig_shape  # (h, w)

        # Ultralytics result timing is reported in milliseconds.
        # result.speed = {'preprocess': ms, 'inference': ms, 'postprocess': ms}
        spd = result.speed
        pre_ms = spd.get('preprocess', 0.0)
        infer_ms = spd.get('inference', 0.0)
        post_ms = spd.get('postprocess', 0.0)
        boxes = result.boxes
        if boxes is not None and len(boxes):
            pred_xyxy = boxes.xyxy.cpu().numpy()  # [M, 4]
            pred_cls = boxes.cls.cpu().numpy().astype(int)  # [M]
            pred_conf = boxes.conf.cpu().numpy()  # [M]
        else:
            pred_xyxy = np.zeros((0, 4))
            pred_cls = np.zeros(0, dtype=int)
            pred_conf = np.zeros(0)

        # Load matching ground-truth label.
        p = Path(image_path)
        label_path = str(p).replace(os.sep + 'images' + os.sep,
                                    os.sep + 'labels' + os.sep)
        label_path = os.path.splitext(label_path)[0] + '.txt'
        gt = load_gt_labels(label_path, img_shape)

        # Compute and merge metrics for this image.
        img_metrics = compute_metrics_per_image(
            pred_xyxy, pred_cls, pred_conf, gt, iou_thres=IOU_THRES
        )
        merge_metrics(global_metrics, img_metrics)

        n_det = len(pred_xyxy)
        print(
            f"[{p.name}]  det: {n_det} "
            f"| pre: {pre_ms:.1f}ms  infer: {infer_ms:.1f}ms  post: {post_ms:.1f}ms  total: {total_ms:.1f}ms "
            f"| mem: {mem_after:.1f}MB ({mem_delta:+.1f}MB) "
            f"| cpu: {cpu_pct:.1f}%"
        )

    all_perf_stats.append({
        'pre_ms': pre_ms,
        'infer_ms': infer_ms,
        'post_ms': post_ms,
        'total_ms': total_ms,
        'mem_mb': mem_after,
        'mem_delta': mem_delta,
        'cpu_pct': cpu_pct,
    })

# Performance summary.
if all_perf_stats:
    n = len(all_perf_stats)


    def avg(key): return sum(x[key] for x in all_perf_stats) / n


    print(f"\n{'=' * 60}")
    print(f"  Processed {n} images   total time: {time.time() - t0:.2f}s")
    print(f"  preprocess: {avg('pre_ms'):.1f}ms  inference: {avg('infer_ms'):.1f}ms  "
          f"postprocess: {avg('post_ms'):.1f}ms  end-to-end: {avg('total_ms'):.1f}ms  (average)")
    print(f"  memory: {avg('mem_mb'):.1f}MB   CPU: {avg('cpu_pct'):.1f}%  (average)")
    print(f"{'=' * 60}")

# Accuracy summary.
if global_metrics:
    print_metrics_table(global_metrics, names, IOU_THRES)




#640
# ============================================================
#   Processed 321   total time: 54.81s
#   preprocess: 1.8ms  inference: 156.7ms  postprocess: 0.6ms  end-to-end: 170.1ms  (average)
#   memory: 598.2MB   CPU: 0.0%  (average)
# ============================================================
#
# ==========================================================================
#   Accuracy evaluation results  (IoU threshold = 0.5)
# ==========================================================================
#   Class                    TP     FP     FN     Prec   Recall       F1       AP
#   ------------------------------------------------------------------------
#   bird                     46     19     57    0.708    0.447    0.548    0.433
#   cat                      49      8     27    0.860    0.645    0.737    0.611
#   cow                      29     21     23    0.580    0.558    0.569    0.476
#   dog                      51     22     38    0.699    0.573    0.630    0.501
#   horse                    66      8     35    0.892    0.653    0.754    0.621
#   sheep                    27      5     41    0.844    0.397    0.540    0.352
#   ------------------------------------------------------------------------
#   ALL                     268     83    221    0.764    0.548    0.638    0.499   mAP
# ==========================================================================

#320
# ============================================================
#   Processed 321   total time: 28.26s
#   preprocess: 0.8ms  inference: 74.1ms  postprocess: 0.5ms  end-to-end: 87.3ms  (average)
#   memory: 557.9MB   CPU: 0.0%  (average)
# ============================================================
#
# ==========================================================================
#   Accuracy evaluation results  (IoU threshold = 0.5)
# ==========================================================================
#   Class                    TP     FP     FN     Prec   Recall       F1       AP
#   ------------------------------------------------------------------------
#   bird                     40     20     63    0.667    0.388    0.491    0.320
#   cat                      42      9     34    0.824    0.553    0.661    0.519
#   cow                      21      8     31    0.724    0.404    0.519    0.390
#   dog                      46     24     43    0.657    0.517    0.579    0.468
#   horse                    65      8     36    0.890    0.644    0.747    0.620
#   sheep                    23      8     45    0.742    0.338    0.465    0.333
#   ------------------------------------------------------------------------
#   ALL                     237     77    252    0.755    0.485    0.590    0.441   mAP
# ==========================================================================

