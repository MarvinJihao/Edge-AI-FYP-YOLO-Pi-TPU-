import cv2
import numpy as np
import torch
import os
from datetime import datetime

from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# -------------------------- 自定义配置参数 --------------------------
MODEL_PATH = "./runs/train/exp9/weights/best.pt"  # 你的模型路径
VIDEO_PATH = "./road_video.mp4"  # 测试视频路径（或0调用摄像头）
SAVE_FOLDER = "./vehicle_detection_results"  # 自定义保存文件夹
VEHICLE_CLASSES = [0,1,2]  # 车辆类别（car, motorcycle, bus, truck）
CONF_THRESH = 0.3  # 置信度阈值
IOU_THRESH = 0.45

  # IOU阈值
FPS = 30  # 视频帧率
PIXEL_PER_METER = 8  # 像素/米比例（根据场景校准）
LINE_POS_RATIO = 0.6  # 测速线位置（画面高度比例）
TARGET_WIDTH = 1280  # 目标显示宽度
TARGET_HEIGHT = 720  # 目标显示高度

# -------------------------- 创建保存文件夹 --------------------------
def create_save_folder():
    """创建保存文件夹（不存在则创建）"""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    # 创建子文件夹
    video_folder = os.path.join(SAVE_FOLDER, "videos")
    data_folder = os.path.join(SAVE_FOLDER, "data")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    return video_folder, data_folder

# -------------------------- 初始化YOLOv5模型 --------------------------
def init_yolov5_model(model_path):
    """加载本地YOLOv5模型"""
    import sys
    sys.path.append("./")  # 本地YOLOv5路径
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    
    device = select_device('cpu')  # 强制使用CPU避免兼容性问题
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    return model, device

# -------------------------- 车辆检测函数 --------------------------
def detect_vehicles(frame, model, device):
    """YOLOv5车辆检测"""
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression
    
    # 预处理帧
    imgsz = (640, 640)
    stride = model.stride
    pt = model.pt
    img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC→CHW, BGR→RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # 推理
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, classes=VEHICLE_CLASSES)
    
    # 解析结果
    detections = []
    for det in pred:
        if len(det):
            det = det.cpu().numpy()
            det[:, :4] = det[:, :4].round()
            for *box, conf, cls in reversed(det):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf.item(), int(cls)))
    return detections

# -------------------------- 速度计算与数据记录类 --------------------------
class SpeedCalculator:
    def __init__(self, fps, pixel_per_meter, data_path):
        self.fps = fps
        self.pixel_per_meter = pixel_per_meter
        self.track_history = defaultdict(list)
        self.cross_time = {}
        
        # 初始化数据记录文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(data_path, f"speed_data_{timestamp}.csv")
        self.csv_file = open(self.log_path, "w", encoding="utf-8")
        self.csv_file.write("TrackID,VehicleType,Speed(km/h),CrossTime,PositionX,PositionY,FrameIndex\n")
        
        # 车辆类型映射
        self.vehicle_types = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    
    def get_vehicle_type(self, cls):
        return self.vehicle_types.get(cls, "Unknown")
    
    def update(self, tracks, frame, frame_idx):
        """更新跟踪并计算速度"""
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 绘制检测框和ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 记录跟踪历史
            self.track_history[track_id].append((center_x, center_y))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            # 测速逻辑
            line_pos = int(frame.shape[0] * LINE_POS_RATIO)
            if abs(center_y - line_pos) < 10:
                if track_id not in self.cross_time:
                    self.cross_time[track_id] = {
                        "time": datetime.now(),
                        "start_x": center_x,
                        "frame": frame_idx
                    }
                else:
                    time_diff = (datetime.now() - self.cross_time[track_id]["time"]).total_seconds()
                    if time_diff > 0.1:
                        pixel_dist = abs(center_x - self.cross_time[track_id]["start_x"])
                        meter_dist = pixel_dist / self.pixel_per_meter
                        speed_mps = meter_dist / time_diff
                        speed_kmh = speed_mps * 3.6
                        
                        # 绘制速度
                        cv2.putText(frame, f"{int(speed_kmh)} km/h", 
                                    (center_x, center_y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 记录数据到CSV
                        vehicle_type = self.get_vehicle_type(track.get_det_class())
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.csv_file.write(
                            f"{track_id},"
                            f"{vehicle_type},"
                            f"{round(speed_kmh, 2)},"
                            f"{current_time},"
                            f"{center_x},"
                            f"{center_y},"
                            f"{frame_idx}\n"
                        )
                        self.csv_file.flush()
                        
                        del self.cross_time[track_id]
            
            # 绘制轨迹
            if len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], np.int32)
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
        
        # 绘制测速线和信息
        line_pos = int(frame.shape[0] * LINE_POS_RATIO)
        cv2.line(frame, (0, line_pos), (frame.shape[1], line_pos), (0, 0, 255), 2)
        cv2.putText(frame, "Speed Detection Line", (10, line_pos-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx} | Tracks: {len(tracks)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame
    
    def close(self):
        self.csv_file.close()

# -------------------------- 主函数 --------------------------
def main():
    # 创建保存文件夹
    video_folder, data_folder = create_save_folder()
    print(f"结果将保存到：{SAVE_FOLDER}")
    
    # 初始化模型
    print("加载YOLOv5模型...")
    model, device = init_yolov5_model(MODEL_PATH)
    
    # 初始化跟踪器和速度计算器
    tracker = DeepSort(max_age=40, n_init=2, nn_budget=100)
    speed_calc = SpeedCalculator(FPS, PIXEL_PER_METER, data_folder)
    
    # 打开视频
    print("打开视频源...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("无法打开视频源！")
        return
    
    # 获取视频信息
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 初始化视频写入器
    video_filename = f"annotated_video_{timestamp}.mp4"
    video_path = os.path.join(video_folder, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, orig_fps, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # 处理视频帧
    frame_idx = 0
    print("开始处理视频...（按Q退出）")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整帧大小
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_idx += 1
        
        # 车辆检测
        detections = detect_vehicles(frame, model, device)
        
        # 更新跟踪器
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # 计算速度并绘制标注
        annotated_frame = speed_calc.update(tracks, frame, frame_idx)
        
        # 保存帧到视频
        video_writer.write(annotated_frame)
        
        # 显示画面
        cv2.imshow("Vehicle Detection & Speed Calculation", annotated_frame)
        
        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    video_writer.release()
    speed_calc.close()
    cv2.destroyAllWindows()
    
    print(f"\n处理完成！")
    print(f"标注视频保存至：{video_path}")
    print(f"速度数据保存至：{speed_calc.log_path}")
    print(f"所有结果位于：{SAVE_FOLDER}")

if __name__ == "__main__":
    main()