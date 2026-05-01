import argparse
import os
import platform
import sys
import time
import psutil
from pathlib import Path

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,
                           box_iou)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


# ═══════════════════════════════════════════════════════════════
#  性能监控工具
# ═══════════════════════════════════════════════════════════════

def get_memory_usage():
    """当前进程 RSS 内存 (MB)"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_cpu_usage():
    """当前进程 CPU 占用率 (%)"""
    return psutil.Process(os.getpid()).cpu_percent(interval=None)


# ═══════════════════════════════════════════════════════════════
#  精度计算工具
# ═══════════════════════════════════════════════════════════════

def load_gt_labels(label_path, img_shape):
    """
    读取 YOLO 格式 gt label，返回 tensor [N, 5] (cls, x1, y1, x2, y2) 像素坐标
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
    return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))


def compute_metrics_per_image(det, gt, iou_thres=0.5):
    """
    单张图 TP/FP/FN 统计。
    det: tensor [M, 6]  (x1,y1,x2,y2, conf, cls) 已映射回原图尺寸
    gt : tensor [N, 5]  (cls, x1,y1,x2,y2)
    返回: {cls_id: {'tp', 'fp', 'fn', 'conf_list'}}
    """
    result = {}
    if len(det) == 0 and len(gt) == 0:
        return result

    pred_boxes = det[:, :4]
    pred_cls = det[:, 5].int()
    pred_conf = det[:, 4]
    gt_boxes = gt[:, 1:]
    gt_cls = gt[:, 0].int()
    gt_matched = torch.zeros(len(gt), dtype=torch.bool)

    for m in range(len(det)):
        pc = pred_cls[m].item()
        conf = pred_conf[m].item()
        if pc not in result:
            result[pc] = {'tp': 0, 'fp': 0, 'fn': 0, 'conf_list': []}

        same_cls_mask = (gt_cls == pc)
        if same_cls_mask.sum() == 0:
            result[pc]['fp'] += 1
            result[pc]['conf_list'].append((conf, 0))
            continue

        ious = box_iou(pred_boxes[m].unsqueeze(0), gt_boxes[same_cls_mask])
        max_iou, max_idx_local = ious[0].max(0)
        global_indices = same_cls_mask.nonzero(as_tuple=False).squeeze(1)
        max_idx_global = global_indices[max_idx_local].item()

        if max_iou >= iou_thres and not gt_matched[max_idx_global]:
            result[pc]['tp'] += 1
            result[pc]['conf_list'].append((conf, 1))
            gt_matched[max_idx_global] = True
        else:
            result[pc]['fp'] += 1
            result[pc]['conf_list'].append((conf, 0))

    # 未命中的 GT → FN
    for n in range(len(gt)):
        if not gt_matched[n]:
            gc = gt_cls[n].item()
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
    """11-point 插值 AP"""
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
    print(f"  精度评估结果  (IoU threshold = {iou_thres})")
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
        cls_name = names[cls_id] if cls_id < len(names) else str(cls_id)
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
        f"  {'ALL':<20} {all_tp:>6} {all_fp:>6} {all_fn:>6} {all_prec:>8.3f} {all_recall:>8.3f} {all_f1:>8.3f} {mAP:>8.3f}  ← mAP")
    print(f"{'=' * 74}\n")


# ═══════════════════════════════════════════════════════════════
#  主推理函数
# ═══════════════════════════════════════════════════════════════

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

    # ── 全局统计容器 ──────────────────────────────────────────
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    all_perf_stats = []  # 性能
    global_metrics = {}  # 精度
    get_cpu_usage()  # 预热 cpu_percent 采样
    # ──────────────────────────────────────────────────────────

    for path, im, im0s, vid_cap, s in dataset:

        # ── 性能：本张图开始 ──────────────────────────────────
        mem_before = get_memory_usage()
        t_img_start = time.perf_counter()
        # ──────────────────────────────────────────────────────

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            visualize_ = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # ── 精度：读取对应 gt label ───────────────────────
            label_path = str(p).replace(os.sep + 'images' + os.sep,
                                        os.sep + 'labels' + os.sep)
            label_path = os.path.splitext(label_path)[0] + '.txt'
            gt = load_gt_labels(label_path, im0.shape)  # [N,5]
            # ──────────────────────────────────────────────────

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:
                        c_ = int(cls)
                        label = None if hide_labels else (names[c_] if hide_conf else f'{names[c_]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c_, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[int(cls)] / f'{p.stem}.jpg', BGR=True)

            # ── 精度：计算并合并本张图指标 ───────────────────
            img_metrics = compute_metrics_per_image(
                det.cpu() if len(det) else torch.zeros((0, 6)),
                gt,
                iou_thres=iou_thres
            )
            merge_metrics(global_metrics, img_metrics)
            # ──────────────────────────────────────────────────

            # Stream / save
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # ── 性能：本张图结束，采集并打印 ─────────────────────
        t_img_end = time.perf_counter()
        mem_after = get_memory_usage()
        cpu_pct = get_cpu_usage()

        pre_ms = dt[0].dt * 1e3  # 预处理
        infer_ms = dt[1].dt * 1e3  # 纯推理
        nms_ms = dt[2].dt * 1e3  # NMS
        total_ms = (t_img_end - t_img_start) * 1e3
        mem_delta = mem_after - mem_before

        LOGGER.info(
            f"{s}"
            f"| pre: {pre_ms:.1f}ms  infer: {infer_ms:.1f}ms  nms: {nms_ms:.1f}ms  total: {total_ms:.1f}ms "
            f"| mem: {mem_after:.1f}MB (Δ{mem_delta:+.1f}MB) "
            f"| cpu: {cpu_pct:.1f}%"
        )
        all_perf_stats.append({
            'pre_ms': pre_ms,
            'infer_ms': infer_ms,
            'nms_ms': nms_ms,
            'total_ms': total_ms,
            'mem_mb': mem_after,
            'mem_delta': mem_delta,
            'cpu_pct': cpu_pct,
        })
        # ──────────────────────────────────────────────────────

    # ── 汇总性能 ──────────────────────────────────────────────
    if all_perf_stats:
        n = len(all_perf_stats)

        def avg(key): return sum(x[key] for x in all_perf_stats) / n

        t = tuple(x.t / seen * 1e3 for x in dt)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        LOGGER.info(
            f"\n{'=' * 60}\n"
            f"  共处理 {n} 张  总推理统计 (均值/张):\n"
            f"  预处理: {avg('pre_ms'):.1f}ms  推理: {avg('infer_ms'):.1f}ms  "
            f"NMS: {avg('nms_ms'):.1f}ms  端到端: {avg('total_ms'):.1f}ms\n"
            f"  内存: {avg('mem_mb'):.1f}MB  CPU: {avg('cpu_pct'):.1f}%\n"
            f"{'=' * 60}"
        )
    # ──────────────────────────────────────────────────────────

    # ── 汇总精度 ──────────────────────────────────────────────
    if global_metrics:
        print_metrics_table(global_metrics, names, iou_thres)
    # ──────────────────────────────────────────────────────────

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp5/weights/best.pt')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default=ROOT / 'runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=3, type=int)
    parser.add_argument('--hide-labels', default=False, action='store_true')
    parser.add_argument('--hide-conf', default=False, action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
