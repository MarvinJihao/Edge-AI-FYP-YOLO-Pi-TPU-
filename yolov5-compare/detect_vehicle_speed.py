import os
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------- Custom configuration --------------------------
MODEL_PATH = "./runs/train/exp9/weights/best.pt"  # Trained model path.
VIDEO_PATH = "./road_video.mp4"  # Input video path. Use 0 for webcam input.
SAVE_FOLDER = "./vehicle_detection_results"  # Output directory.
VEHICLE_CLASSES = [0, 1, 2]  # Vehicle class IDs used by the trained model.
CONF_THRESH = 0.3  # Confidence threshold.
IOU_THRESH = 0.45  # NMS IoU threshold.
FPS = 30  # Video frame rate.
PIXEL_PER_METER = 8  # Pixel-to-meter ratio. Calibrate for the target scene.
LINE_POS_RATIO = 0.6  # Speed line position as a ratio of frame height.
TARGET_WIDTH = 1280  # Output display width.
TARGET_HEIGHT = 720  # Output display height.


# -------------------------- Output folders --------------------------
def create_save_folder():
    """Create output folders if they do not already exist."""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # Create separate folders for rendered videos and measurement data.
    video_folder = os.path.join(SAVE_FOLDER, "videos")
    data_folder = os.path.join(SAVE_FOLDER, "data")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    return video_folder, data_folder


# -------------------------- YOLOv5 model setup --------------------------
def init_yolov5_model(model_path):
    """Load the local YOLOv5 model."""
    import sys

    sys.path.append("./")  # Local YOLOv5 source path.
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    # Use CPU to avoid device compatibility issues on deployment machines.
    device = select_device('cpu')
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    return model, device


# -------------------------- Vehicle detection --------------------------
def detect_vehicles(frame, model, device):
    """Run YOLOv5 vehicle detection on one frame."""
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression

    # Preprocess frame.
    imgsz = (640, 640)
    stride = model.stride
    pt = model.pt
    img = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # Convert HWC to CHW and BGR to RGB.
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run inference and non-max suppression.
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, classes=VEHICLE_CLASSES)

    # Convert detections into DeepSORT input format.
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


# -------------------------- Speed calculation and data logging --------------------------
class SpeedCalculator:
    def __init__(self, fps, pixel_per_meter, data_path):
        self.fps = fps
        self.pixel_per_meter = pixel_per_meter
        self.track_history = defaultdict(list)
        self.cross_time = {}

        # Create the CSV log file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(data_path, f"speed_data_{timestamp}.csv")
        self.csv_file = open(self.log_path, "w", encoding="utf-8")
        self.csv_file.write("TrackID,VehicleType,Speed(km/h),CrossTime,PositionX,PositionY,FrameIndex\n")

        # Class ID to display-name mapping. Adjust to match the custom dataset.
        self.vehicle_types = {0: "Car", 1: "Motorcycle", 2: "Bus", 3: "Truck"}

    def get_vehicle_type(self, cls):
        return self.vehicle_types.get(cls, "Unknown")

    def update(self, tracks, frame, frame_idx):
        """Update tracks, estimate speed, and draw annotations."""
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw the bounding box and track ID.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # Store recent track positions.
            self.track_history[track_id].append((center_x, center_y))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Estimate speed when the tracked object crosses the speed line.
            line_pos = int(frame.shape[0] * LINE_POS_RATIO)
            if abs(center_y - line_pos) < 10:
                if track_id not in self.cross_time:
                    self.cross_time[track_id] = {
                        "time": datetime.now(),
                        "start_x": center_x,
                        "frame": frame_idx,
                    }
                else:
                    time_diff = (datetime.now() - self.cross_time[track_id]["time"]).total_seconds()
                    if time_diff > 0.1:
                        pixel_dist = abs(center_x - self.cross_time[track_id]["start_x"])
                        meter_dist = pixel_dist / self.pixel_per_meter
                        speed_mps = meter_dist / time_diff
                        speed_kmh = speed_mps * 3.6

                        # Draw estimated speed.
                        cv2.putText(
                            frame,
                            f"{int(speed_kmh)} km/h",
                            (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        # Write speed data to CSV.
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

            # Draw the recent movement path.
            if len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], np.int32)
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

        # Draw the speed line and frame information.
        line_pos = int(frame.shape[0] * LINE_POS_RATIO)
        cv2.line(frame, (0, line_pos), (frame.shape[1], line_pos), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "Speed Detection Line",
            (10, line_pos - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_idx} | Tracks: {len(tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        return frame

    def close(self):
        self.csv_file.close()


# -------------------------- Main entry point --------------------------
def main():
    # Create output folders.
    video_folder, data_folder = create_save_folder()
    print(f"Results will be saved to: {SAVE_FOLDER}")

    # Load the detection model.
    print("Loading YOLOv5 model...")
    model, device = init_yolov5_model(MODEL_PATH)

    # Initialise tracker and speed calculator.
    tracker = DeepSort(max_age=40, n_init=2, nn_budget=100)
    speed_calc = SpeedCalculator(FPS, PIXEL_PER_METER, data_folder)

    # Open the input video.
    print("Opening video source...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Unable to open video source.")
        return

    # Read video metadata.
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the output video writer.
    video_filename = f"annotated_video_{timestamp}.mp4"
    video_path = os.path.join(video_folder, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, orig_fps, (TARGET_WIDTH, TARGET_HEIGHT))

    # Process video frames.
    frame_idx = 0
    print("Processing video... Press Q to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for display and output.
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_idx += 1

        # Run vehicle detection.
        detections = detect_vehicles(frame, model, device)

        # Update tracker.
        tracks = tracker.update_tracks(detections, frame=frame)

        # Estimate speed and draw annotations.
        annotated_frame = speed_calc.update(tracks, frame, frame_idx)

        # Save frame to output video.
        video_writer.write(annotated_frame)

        # Display the current frame.
        cv2.imshow("Vehicle Detection & Speed Calculation", annotated_frame)

        # Exit when Q is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources.
    cap.release()
    video_writer.release()
    speed_calc.close()
    cv2.destroyAllWindows()

    print("\nProcessing complete.")
    print(f"Annotated video saved to: {video_path}")
    print(f"Speed data saved to: {speed_calc.log_path}")
    print(f"All results are located in: {SAVE_FOLDER}")


if __name__ == "__main__":
    main()
