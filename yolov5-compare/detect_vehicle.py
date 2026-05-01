import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 将YOLOv5代码包路径添加到系统环境（关键修改）
yolov5_path = r'E:\pycharm_workspace\yolov5-7.0-1'  # 你的本地YOLOv5路径
sys.path.append(yolov5_path)

# 从本地加载YOLOv5模型（关键修改）
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# 设置路径
model_path = r'E:\pycharm_workspace\yolov5-7.0-1\runs\train\exp9\weights\best.pt'
image_dir = r'E:\pycharm_workspace\yolov5-7.0-1\picture'
output_dir = r'E:\pycharm_workspace\yolov5-7.0-1\ratio'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 本地加载模型配置（关键修改）
device = select_device('')  # 自动选择CPU/GPU
model = DetectMultiBackend(model_path, device=device, dnn=False, data=os.path.join(yolov5_path, 'data/vehicle1.yaml'))
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # 检查图片尺寸

# 设置模型参数
conf_thres = 0.3 # 置信度阈值
iou_thres = 0.45   # NMS IoU阈值
max_det = 1000     # 最大检测数

# 获取图片列表
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() in image_extensions]

# 存储结果的数据列表
results_data = []

print(f"找到 {len(image_files)} 张图片，开始处理...")

for image_file in image_files:
    try:
        # 构建完整路径
        image_path = os.path.join(image_dir, image_file)
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_file}")
            continue
        
        # 获取图片尺寸和总面积
        height, width = img.shape[:2]
        total_area = height * width
        
        # 预处理图片（适应模型输入）
        img_copy = img.copy()
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).to(device)
        img_tensor = img_tensor.half() if model.fp16 else img_tensor.float()  # uint8 to fp16/32
        img_tensor /= 255.0  # 归一化到0-1
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[None]  # 扩展batch维度
        
        # 运行检测
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        
        # 处理检测结果
        vehicle_count = 0
        vehicle_area = 0
        annotator = Annotator(img_copy, line_width=2, example=str(names))
        
        # 筛选车辆类别（根据你的模型训练情况调整类别索引/名称）
        vehicle_class_ids = [0, 1, 2,]  # COCO数据集：2=car,3=motorcycle,5=bus,7=truck
        # 如果你的模型用自定义类别名称，可改为：
        # vehicle_class_names = ['car', 'motorcycle', 'bus', 'truck', 'vehicle']
        # vehicle_class_ids = [names.index(name) for name in vehicle_class_names if name in names]
        
        for det in pred:
            if len(det):
                # 将检测框缩放到原图尺寸
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()
                
                # 处理每个检测框
                for *xyxy, conf, cls in reversed(det):
                    cls_id = int(cls)
                    
                    # 筛选车辆类别
                    if cls_id in vehicle_class_ids:
                        vehicle_count += 1
                        
                        # 计算车辆面积
                        x1, y1, x2, y2 = map(int, xyxy)
                        area = (x2 - x1) * (y2 - y1)
                        vehicle_area += area
                        
                        # 绘制检测框和标签
                        c = int(cls)  # 类别ID
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
        
        # 生成带检测框的图片
        img_with_boxes = annotator.result()
        
        # 计算占比
        vehicle_area_ratio = (vehicle_area / total_area) * 100 if total_area > 0 else 0
        empty_area_ratio = 100 - vehicle_area_ratio
        
        # 保存带检测框的图片
        output_image_path = os.path.join(output_dir, f"detected_{image_file}")
        cv2.imwrite(output_image_path, img_with_boxes)
        
        # 存储结果
        result = {
            '图片名称': image_file,
            '车辆数量': vehicle_count,
            '图片总面积(像素)': total_area,
            '车辆总面积(像素)': vehicle_area,
            '车辆面积占比(%)': round(vehicle_area_ratio, 2),
            '空地面积占比(%)': round(empty_area_ratio, 2),
            '检测图片路径': output_image_path
        }
        results_data.append(result)
        
        print(f"处理完成: {image_file} - 车辆数: {vehicle_count}, 车辆占比: {vehicle_area_ratio:.2f}%")
        
    except Exception as e:
        print(f"处理图片 {image_file} 时出错: {str(e)}")
        continue

# 保存结果到CSV文件
if results_data:
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, '车辆检测统计结果.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 保存结果到Excel文件（如果安装了openpyxl）
    try:
        excel_path = os.path.join(output_dir, '车辆检测统计结果.xlsx')
        results_df.to_excel(excel_path, index=False, engine='openpyxl')
    except ImportError:
        print("未安装openpyxl，跳过Excel文件保存")
    
    print(f"\n结果已保存到:")
    print(f"CSV文件: {csv_path}")
    if 'excel_path' in locals():
        print(f"Excel文件: {excel_path}")
    
    # 打印汇总统计
    total_vehicles = sum([r['车辆数量'] for r in results_data])
    avg_vehicle_ratio = np.mean([r['车辆面积占比(%)'] for r in results_data])
    print(f"\n汇总统计:")
    print(f"处理图片总数: {len(results_data)}")
    print(f"检测到车辆总数: {total_vehicles}")
    print(f"平均车辆面积占比: {avg_vehicle_ratio:.2f}%")
else:
    print("没有处理任何图片")

print("\n处理完成！")