import os
import yaml
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 1. 配置参数（按需修改）
# ============================================================
CONFIG = {
    # 模型选择: yolov8n/s/m/l/x (n最小最快, x最大最准)
    "model": "yolov8n.pt",

    # 数据集配置文件路径（自定义数据集需修改）
    "data": "CrowdHumanHead/CrowdHumanHead.yaml",
    # 训练超参数
    "epochs": 1,
    "batch": 16,
    "imgsz": 640,
    "lr0": 0.01,  # 初始学习率
    "lrf": 0.01,  # 最终学习率 = lr0 * lrf
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,

    # 数据增强
    "augment": True,
    "hsv_h": 0.015,  # 色调增强
    "hsv_s": 0.7,  # 饱和度增强
    "hsv_v": 0.4,  # 明度增强
    "flipud": 0.0,  # 上下翻转概率
    "fliplr": 0.5,  # 左右翻转概率
    "mosaic": 1.0,  # Mosaic增强概率

    # 训练设置
    "device": "",  # "" 自动选择, 0=GPU, "cpu"=CPU
    "workers": 8,  # 数据加载线程数（Windows建议设为0）
    "patience": 50,  # 早停轮数（验证集无提升时停止）
    "project": "runs/train",
    "name": "yolov8_exp",
    "exist_ok": False,  # 是否覆盖已有实验
    # 导出设置
    "export_format": "torchscript",  # 导出格式: onnx/tflite/torchscript
}

def create_dataset_yaml(
        train_path: str,
        val_path: str,
        class_names: list,
        save_path: str = "custom_dataset.yaml"
):
    """
    生成自定义数据集的YAML配置文件

    数据集目录结构应为:
    dataset/
        images/
            train/  *.jpg
            val/    *.jpg
        labels/
            train/  *.txt  (YOLO格式标注)
            val/    *.txt

    YOLO标注格式（每行）:
        class_id cx cy w h
        （归一化坐标，相对于图像宽高）
    """
    data = {
        "path": str(Path(train_path).parent.parent),  # 数据集根目录
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"数据集配置已生成: {save_path}")
    return save_path


# ============================================================
# 3. 训练函数
# ============================================================
def train(config: dict):
    print("=" * 50)
    print("开始训练 YOLOv8 模型")
    print("=" * 50)

    # 加载模型
    model = YOLO(config["model"])
    print(f"模型: {config['model']}")
    print(f"数据集: {config['data']}")
    print(f"训练轮数: {config['epochs']}, 批次: {config['batch']}, 图像尺寸: {config['imgsz']}")
    print("-" * 50)

    # 开始训练
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
        plots=True,  # 生成训练曲线图
        save=True,  # 保存checkpoint
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("\n" + "=" * 50)
    print(f"训练完成！")
    print(f"结果保存路径: {save_dir}")
    print(f"最佳模型: {save_dir / 'weights/best.pt'}")
    print(f"最终模型: {save_dir / 'weights/last.pt'}")
    return model, save_dir


# ============================================================
# 4. 验证函数
# ============================================================
def validate(model, data: str):
    print("\n" + "=" * 50)
    print("📊 开始验证模型...")
    metrics = model.val(data=data)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    return metrics


# ============================================================
# 5. 推理测试函数
# ============================================================
def predict(model, source: str, conf: float = 0.25, save: bool = True):
    """
    source: 图片路径 / 视频路径 / 摄像头(0) / URL
    """
    print("\n" + "=" * 50)
    print(f"推理测试: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=False,  # 树莓派无显示器时设为False
    )
    for r in results:
        print(f"  检测到 {len(r.boxes)} 个目标")
    return results


def export_model(model_path: str, export_format: str = "onnx"):
    """
    导出模型用于部署

    常用格式:
      onnx       - 通用推理框架
      tflite     - 树莓派/移动端 (加 int8=True 量化)
      torchscript - PyTorch部署
      ncnn       - 移动端高性能
    """
    print("\n" + "=" * 50)
    print(f"导出模型为 {export_format.upper()} 格式...")
    model = YOLO(model_path)

    if export_format == "tflite":
        # INT8量化，适合树莓派CPU推理
        path = model.export(format="tflite", int8=True, imgsz=640)
    elif export_format == "onnx":
        path = model.export(format="onnx", dynamic=False, simplify=True)
    else:
        path = model.export(format=export_format)

    print(f"导出完成: {path}")
    return path


if __name__ == "__main__":
    # 如果使用自己的数据集，取消注释并修改以下内容：
    #
    # dataset_yaml = create_dataset_yaml(
    #     train_path="dataset/images/train",
    #     val_path="dataset/images/val",
    #     class_names=["cat", "dog", "person"],  # 替换为你的类别
    #     save_path="custom_dataset.yaml"
    # )
    # CONFIG["data"] = dataset_yaml
    # ──────────────────────────────────────────────────────

    # ── 训练 ──────────────────────────────────────────────
    model, save_dir = train(CONFIG)

    # ── 验证 ──────────────────────────────────────────────
    validate(model, CONFIG["data"])

    # ── 推理测试（替换为你的测试图片）──────────────────────
    # predict(model, source="test.jpg", conf=0.25)

    # ── 导出模型（用于树莓派部署）──────────────────────────
    best_model_path = str(save_dir / "weights/best.pt")
    export_model(best_model_path, export_format=CONFIG["export_format"])

    print("\n全部流程完成！")