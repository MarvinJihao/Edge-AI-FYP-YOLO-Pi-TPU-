from ultralytics import YOLO
import os

weight_path =r"./runs/detect/train3 640/weights/best.pt"
# Load a model
model = YOLO(weight_path)  # load a pretrained model (recommended for training)
# image folder path
image_folder = "datasets/animal_yolo_format/images/test"

# collect image paths in the folder
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

# inference
for image_path in image_paths:
    # results = model([image_path], device=[7])  # use the selected deviceinference
    # for result in results:
        # boxes = result.boxes  # Boxes object for bounding box outputs
        # result.save(filename=f"result_{os.path.basename(image_path)}")  #
    results = model.predict(task="detect", source=image_path, imgsz=640, max_det=1000, conf=0.60, show_labels=False, show_conf=True, save=True, device="cpu", augment=True)
