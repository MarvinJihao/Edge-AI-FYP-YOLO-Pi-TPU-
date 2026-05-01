import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# Add the local YOLOv5 source directory to the Python path.
yolov5_path = r'E:\pycharm_workspace\yolov5-7.0-1'
sys.path.append(yolov5_path)

# Load YOLOv5 modules from the local source tree.
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Input and output paths.
model_path = r'E:\pycharm_workspace\yolov5-7.0-1\runs\train\exp9\weights\best.pt'
image_dir = r'E:\pycharm_workspace\yolov5-7.0-1\picture'
output_dir = r'E:\pycharm_workspace\yolov5-7.0-1\ratio'

# Create the output directory if it does not already exist.
os.makedirs(output_dir, exist_ok=True)

# Load the trained model.
device = select_device('')
model = DetectMultiBackend(model_path, device=device, dnn=False, data=os.path.join(yolov5_path, 'data/vehicle1.yaml'))
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)

# Detection settings.
conf_thres = 0.3
iou_thres = 0.45
max_det = 1000

# Collect supported image files.
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() in image_extensions]

# Store per-image statistics.
results_data = []

print(f"Found {len(image_files)} images. Starting processing...")

for image_file in image_files:
    try:
        # Build the full input path.
        image_path = os.path.join(image_dir, image_file)

        # Read the input image.
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unable to read image: {image_file}")
            continue

        # Get image dimensions and total area.
        height, width = img.shape[:2]
        total_area = height * width

        # Preprocess the image for model inference.
        img_copy = img.copy()
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).to(device)
        img_tensor = img_tensor.half() if model.fp16 else img_tensor.float()
        img_tensor /= 255.0
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[None]

        # Run detection.
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        # Process detection results.
        vehicle_count = 0
        vehicle_area = 0
        annotator = Annotator(img_copy, line_width=2, example=str(names))

        # Vehicle class IDs should match the custom training dataset.
        vehicle_class_ids = [0, 1, 2]
        # If the model uses class names instead, this can be adapted as:
        # vehicle_class_names = ['car', 'motorcycle', 'bus', 'truck', 'vehicle']
        # vehicle_class_ids = [names.index(name) for name in vehicle_class_names if name in names]

        for det in pred:
            if len(det):
                # Scale detection boxes back to the original image size.
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()

                # Process each detection box.
                for *xyxy, conf, cls in reversed(det):
                    cls_id = int(cls)

                    # Keep vehicle classes only.
                    if cls_id in vehicle_class_ids:
                        vehicle_count += 1

                        # Calculate vehicle bounding-box area.
                        x1, y1, x2, y2 = map(int, xyxy)
                        area = (x2 - x1) * (y2 - y1)
                        vehicle_area += area

                        # Draw the detection box and label.
                        c = int(cls)
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

        # Generate the annotated image.
        img_with_boxes = annotator.result()

        # Calculate the vehicle-area ratio.
        vehicle_area_ratio = (vehicle_area / total_area) * 100 if total_area > 0 else 0
        empty_area_ratio = 100 - vehicle_area_ratio

        # Save the annotated image.
        output_image_path = os.path.join(output_dir, f"detected_{image_file}")
        cv2.imwrite(output_image_path, img_with_boxes)

        # Store statistics for this image.
        result = {
            'image_name': image_file,
            'vehicle_count': vehicle_count,
            'total_image_area_pixels': total_area,
            'vehicle_area_pixels': vehicle_area,
            'vehicle_area_ratio_percent': round(vehicle_area_ratio, 2),
            'empty_area_ratio_percent': round(empty_area_ratio, 2),
            'annotated_image_path': output_image_path,
        }
        results_data.append(result)

        print(
            f"Processed: {image_file} - vehicles: {vehicle_count}, "
            f"vehicle area ratio: {vehicle_area_ratio:.2f}%"
        )

    except Exception as e:
        print(f"Error processing image {image_file}: {str(e)}")
        continue

# Save results to CSV and Excel files.
if results_data:
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, 'vehicle_detection_statistics.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Save results to Excel if openpyxl is available.
    try:
        excel_path = os.path.join(output_dir, 'vehicle_detection_statistics.xlsx')
        results_df.to_excel(excel_path, index=False, engine='openpyxl')
    except ImportError:
        print("openpyxl is not installed. Skipping Excel export.")

    print("\nResults saved:")
    print(f"CSV file: {csv_path}")
    if 'excel_path' in locals():
        print(f"Excel file: {excel_path}")

    # Print summary statistics.
    total_vehicles = sum([r['vehicle_count'] for r in results_data])
    avg_vehicle_ratio = np.mean([r['vehicle_area_ratio_percent'] for r in results_data])
    print("\nSummary statistics:")
    print(f"Total processed images: {len(results_data)}")
    print(f"Total detected vehicles: {total_vehicles}")
    print(f"Average vehicle area ratio: {avg_vehicle_ratio:.2f}%")
else:
    print("No images were processed.")

print("\nProcessing complete.")
