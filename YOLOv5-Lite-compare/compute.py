import argparse
import time
import os
import psutil
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, box_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_cpu_usage():
    process = psutil.Process(os.getpid())
    return process.cpu_percent(interval=None)


#  Accuracy metric helpers

def load_gt_labels(label_path, img_shape):
    """
    Load YOLO ground-truth labels as tensor [N, 5] (cls, x1, y1, x2, y2).
    label_path: str, img_shape: (h, w)
    """
    h, w = img_shape[:2]
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(
                    parts[4])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                labels.append([cls, x1, y1, x2, y2])
    return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))


def compute_metrics_per_image(det, gt, iou_thres=0.5, nc=None):
    """
    Compute TP, FP, and FN values for one image.
    det: tensor [M, 6] (x1, y1, x2, y2, conf, cls)
    gt : tensor [N, 5] (cls, x1, y1, x2, y2)
    Returns: {cls_id: {'tp':, 'fp':, 'fn':, 'conf_list':[]}}
    """
    result = {}

    pred_boxes = det[:, :4]  # [M,4]
    pred_cls = det[:, 5].int()
    pred_conf = det[:, 4]

    gt_boxes = gt[:, 1:]  # [N,4]
    gt_cls = gt[:, 0].int()

    gt_matched = torch.zeros(len(gt), dtype=torch.bool)

    for m in range(len(det)):
        pc = pred_cls[m].item()
        conf = pred_conf[m].item()

        if pc not in result:
            result[pc] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}

        # Match each prediction against ground-truth boxes of the same class.
        same_cls_mask = (gt_cls == pc)
        if same_cls_mask.sum() == 0:
            result[pc]['fp'] += 1
            result[pc]['conf_list'].append((conf, 0))  # (conf, is_tp)
            continue

        ious = box_iou(pred_boxes[m].unsqueeze(0), gt_boxes[same_cls_mask])  # [1, K]
        max_iou, max_idx_local = ious[0].max(0)

        # Map the local match index back to the global ground-truth index.
        global_indices = same_cls_mask.nonzero(as_tuple=False).squeeze(1)
        max_idx_global = global_indices[max_idx_local].item()

        if max_iou >= iou_thres and not gt_matched[max_idx_global]:
            result[pc]['tp'] += 1
            result[pc]['conf_list'].append((conf, 1))
            gt_matched[max_idx_global] = True
        else:
            result[pc]['fp'] += 1
            result[pc]['conf_list'].append((conf, 0))

    # Count unmatched ground-truth boxes as false negatives.
    for n in range(len(gt)):
        if not gt_matched[n]:
            gc = gt_cls[n].item()
            if gc not in result:
                result[gc] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}
            result[gc]['fn'] += 1

    return result


def merge_metrics(global_metrics, image_metrics):
    """Merge per-image metrics into the global metrics dictionary."""
    for cls_id, vals in image_metrics.items():
        if cls_id not in global_metrics:
            global_metrics[cls_id] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}
        global_metrics[cls_id]['tp'] += vals['tp']
        global_metrics[cls_id]['fp'] += vals['fp']
        global_metrics[cls_id]['fn'] += vals['fn']
        global_metrics[cls_id]['conf_list'] += vals['conf_list']


def compute_ap(conf_list, total_gt):
    """
    11-point interpolation AP
    conf_list: [(conf, is_tp), ...]
    total_gt: number of ground-truth boxes for this class
    """
    if total_gt == 0 or len(conf_list) == 0:
        return 0.0
    conf_list = sorted(conf_list, key=lambda x: -x[0])
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in conf_list:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    # 11-point
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(prec) if prec else 0.0
    return ap / 11


def print_metrics_table(global_metrics, names, iou_thres):
    """Print per-class and overall Precision / Recall / F1 / AP."""
    print(f"\n{'=' * 70}")
    print(f"  Accuracy evaluation results (IoU threshold = {iou_thres})")
    print(f"{'=' * 70}")
    print(f"  {'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP':>8}")
    print(f"  {'-' * 68}")

    all_tp, all_fp, all_fn = 0, 0, 0
    all_ap = []

    for cls_id, vals in sorted(global_metrics.items()):
        tp, fp, fn = vals['tp'], vals['fp'], vals['fn']
        total_gt = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        ap = compute_ap(vals['conf_list'], total_gt)

        cls_name = names[cls_id] if cls_id < len(names) else str(cls_id)
        print(f"  {cls_name:<20} {tp:>6} {fp:>6} {fn:>6} {prec:>8.3f} {recall:>8.3f} {f1:>8.3f} {ap:>8.3f}")

        all_tp += tp;
        all_fp += fp;
        all_fn += fn
        all_ap.append(ap)

    # Overall metrics.
    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = 2 * all_prec * all_recall / (all_prec + all_recall) if (all_prec + all_recall) > 0 else 0.0
    mAP = sum(all_ap) / len(all_ap) if all_ap else 0.0

    print(f"  {'-' * 68}")
    print(
        f"  {'ALL':<20} {all_tp:>6} {all_fp:>6} {all_fn:>6} {all_prec:>8.3f} {all_recall:>8.3f} {all_f1:>8.3f} {mAP:>8.3f}   mAP")
    print(f"{'=' * 70}\n")



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # Global statistics containers.
    all_perf_stats = []  # performance statistics
    global_metrics = {}  # accuracy statistics

    t0 = time.time()
    get_cpu_usage()  # warm up

    for path, img, im0s, vid_cap in dataset:

        # Start performance sampling.
        mem_before = get_memory_usage()
        t_img_start = time.perf_counter()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            # Load matching ground-truth label:
            # images/test/xxx.jpg -> labels/test/xxx.txt
            img_path = Path(p)
            label_path = str(img_path).replace(os.sep + 'images' + os.sep,
                                               os.sep + 'labels' + os.sep)
            label_path = os.path.splitext(label_path)[0] + '.txt'
            gt = load_gt_labels(label_path, im0.shape)  # [N,5] cls,x1,y1,x2,y2

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Compute and merge metrics for this image.
            img_metrics = compute_metrics_per_image(
                det.cpu() if len(det) else torch.zeros((0, 6)),
                gt,
                iou_thres=opt.iou_thres
            )
            merge_metrics(global_metrics, img_metrics)

            # Finish performance sampling.
            t_img_end = time.perf_counter()
            mem_after = get_memory_usage()
            cpu_pct = get_cpu_usage()
            infer_ms = (t2 - t1) * 1000
            total_ms = (t_img_end - t_img_start) * 1000
            mem_delta = mem_after - mem_before

            print(f"{s}Done. "
                  f"| infer+NMS: {infer_ms:.1f}ms  total: {total_ms:.1f}ms "
                  f"| mem: {mem_after:.1f}MB ({mem_delta:+.1f}MB) "
                  f"| cpu: {cpu_pct:.1f}%")

            all_perf_stats.append({
                'infer_ms': infer_ms,
                'total_ms': total_ms,
                'mem_mb': mem_after,
                'mem_delta': mem_delta,
                'cpu_pct': cpu_pct,
            })
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # Performance summary.
    if all_perf_stats:
        n = len(all_perf_stats)
        avg_infer = sum(x['infer_ms'] for x in all_perf_stats) / n
        avg_total = sum(x['total_ms'] for x in all_perf_stats) / n
        avg_mem = sum(x['mem_mb'] for x in all_perf_stats) / n
        avg_cpu = sum(x['cpu_pct'] for x in all_perf_stats) / n
        print(f"\n{'=' * 60}")
        print(f"Processed {n} images  total time: {time.time() - t0:.3f}s")
        print(f"  inference+NMS average: {avg_infer:.1f}ms   total average: {avg_total:.1f}ms")
        print(f"  memory average: {avg_mem:.1f}MB   CPU average: {avg_cpu:.1f}%")
        print(f"{'=' * 60}")

    # Accuracy summary.
    if global_metrics:
        print_metrics_table(global_metrics, names, opt.iou_thres)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best.pt')
    parser.add_argument('--source', type=str, default='dataset/animal_yolo_format/images/test')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.45)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
