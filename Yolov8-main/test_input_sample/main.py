import os
import yaml
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 1. Configuration parameters (modify as needed)
# ============================================================
CONFIG = {
    # Model selection: yolov8n/s/m/l/x (n is fastest, x is most accurate).
    "model": "yolov8n.pt",

    # Dataset configuration path.
    "data": "CrowdHumanHead/CrowdHumanHead.yaml",
    # training hyperparameters
    "epochs": 1,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,  # initial learning rate
    "lrf": 0.01,  # final learning rate = lr0 * lrf
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,

    # data augmentation
    "augment": True,
    "hsv_h": 0.015,  # hue augmentation
    "hsv_s": 0.7,  # saturation augmentation
    "hsv_v": 0.4,  # value augmentation
    "flipud": 0.0,  # vertical flip probability
    "fliplr": 0.5,  # horizontal flip probability
    "mosaic": 1.0,  # Mosaic

    # training settings
    "device": "",  # "" automatic selection, 0=GPU, "cpu"=CPU
    "workers": 8,  # data loading workers; use 0 on Windows if multiprocessing causes issues
    "patience": 50,  # early stopping patience
    "project": "runs/train",
    "name": "yolov8_exp",
    "exist_ok": False,  # whether to overwrite existing experiments
    # export settings
    "export_format": "torchscript",  # export format: onnx/tflite/torchscript
}

def create_dataset_yaml(
        train_path: str,
        val_path: str,
        class_names: list,
        save_path: str = "custom_dataset.yaml"
):
    """
    Generate a YAML configuration file for a custom dataset

    Expected dataset directory structure:
    dataset/
        images/
            train/  *.jpg
            val/    *.jpg
        labels/
            train/  *.txt  (YOLO)
            val/    *.txt

    YOLO:
        class_id cx cy w h
        normalized coordinates relative to image width and height
    """
    data = {
        "path": str(Path(train_path).parent.parent),  # dataset root
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"Dataset configuration generated: {save_path}")
    return save_path


# ============================================================
# 3. training function
# ============================================================
def train(config: dict):
    print("=" * 50)
    print("Start training YOLOv8 model")
    print("=" * 50)

    # load model
    model = YOLO(config["model"])
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['data']}")
    print(f"Epochs: {config['epochs']}, batch size: {config['batch']}, image size: {config['imgsz']}")
    print("-" * 50)

    # start training
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        augment=config["augment"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        device=config["device"],
        workers=config["workers"],
        patience=config["patience"],
        project=config["project"],
        name=config["name"],
        exist_ok=config["exist_ok"],
        plots=True,  # generate training curves
        save=True,  # save checkpoint
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("\n" + "=" * 50)
    print("Training complete")
    print(f"Results saved to: {save_dir}")
    print(f"Best model: {save_dir / 'weights/best.pt'}")
    print(f"Final model: {save_dir / 'weights/last.pt'}")
    return model, save_dir


# ============================================================
# 4. validation function
# ============================================================
def validate(model, data: str):
    print("\n" + "=" * 50)
    print("Validating model...")
    metrics = model.val(data=data)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    return metrics


# ============================================================
# 5. inference
# ============================================================
def predict(model, source: str, conf: float = 0.25, save: bool = True):
    """
    source: image path / video path / webcam(0) / URL
    """
    print("\n" + "=" * 50)
    print(f"inference: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=False,  # keep False for headless Raspberry Pi runs
    )
    for r in results:
        print(f"  detected {len(r.boxes)} objects")
    return results


def export_model(model_path: str, export_format: str = "onnx"):
    """
    Export a trained model for deployment.

    common formats:
      onnx        - general inference runtime
      tflite      - Raspberry Pi/mobile deployment (use int8=True for quantization)
      torchscript - PyTorch deployment
      ncnn        - high-performance mobile deployment
    """
    print("\n" + "=" * 50)
    print(f"Exporting model as {export_format.upper()}...")
    model = YOLO(model_path)

    if export_format == "tflite":
        # INT8 quantization is suitable for Raspberry Pi CPU inference.
        path = model.export(format="tflite", int8=True, imgsz=640)
    elif export_format == "onnx":
        path = model.export(format="onnx", dynamic=False, simplify=True)
    else:
        path = model.export(format=export_format)

    print(f"export complete: {path}")
    return path


if __name__ == "__main__":
    # Uncomment this block when using a custom dataset.
    # dataset_yaml = create_dataset_yaml(
    #     train_path="dataset/images/train",
    #     val_path="dataset/images/val",
    #     class_names=["cat", "dog", "person"],
    #     save_path="custom_dataset.yaml"
    # )
    # CONFIG["data"] = dataset_yaml

    # Train the model.
    model, save_dir = train(CONFIG)

    # Validate the trained model.
    validate(model, CONFIG["data"])

    #  inference
    # predict(model, source="test.jpg", conf=0.25)

    # Export the best model for deployment.
    best_model_path = str(save_dir / "weights/best.pt")
    export_model(best_model_path, export_format=CONFIG["export_format"])

    print("\nfull workflow complete")
