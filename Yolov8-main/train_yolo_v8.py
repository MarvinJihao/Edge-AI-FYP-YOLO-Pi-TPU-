from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="dataset/CrowdHumanHead.yaml",
            epochs=2000, batch=24, device=[0], imgsz=640)
            #loggers='tensorboard')  # train the model
            #imgsz=640+160=800
metrics = model.val()  # evaluate model performance on the validation set
path = model.export(format="Pytorch")  # export the model to ONNX format
print("Export Path:", path)