from ultralytics import YOLO, settings


settings.update({'datasets_dir': '/root/autodl-tmp/Yolov8-main/datasets'})
# model
model = YOLO('yolov8n.yaml')

# Training the model
results = model.train(data="data/CrowdHumanHead.yaml", epochs=2000 ,imgsz=640)
