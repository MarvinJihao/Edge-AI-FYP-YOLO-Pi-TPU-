#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
import shutil
import time
import onnxruntime as ort
import numpy as np
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QTabWidget
)
from utils.torch_utils import select_device
import cv2
import threading
import os
import sys
from pathlib import Path
import os.path as osp
from PIL import Image, ImageDraw, ImageFont
import time


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs


def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)
    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []


def infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, dsize=(model_w, model_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
    # 输出坐标矫正
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)
    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)
    return boxes, confs, ids


# 定义MainWindow类
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('老人摔倒检测系统')
        self.resize(1200, 700)
        #self.setWindowIcon(QIcon("images/UI/lufei.png"))
        self.setWindowIcon(QPixmap("images/UI/lufei.png"))
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        self.vid_source = '0'
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model, self.model_h, self.model_w, self.nl, self.na, self.stride, self.anchor_grid = self.model_load()
        self.initUI()
        self.reset_vid()

    def model_load(self):
        # 模型加载部分...
        # 模型加载
        model_pb_path = "./best.onnx"
        so = ort.SessionOptions()
        self.net = ort.InferenceSession(model_pb_path, so)


        # 模型参数
        self.model_h = 640
        self.model_w = 640
        self.nl = 3
        self.na = 3
        self.stride = [8., 16., 32.]
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        return self.net, self.model_h, self.model_w, self.nl, self.na, self.stride, self.anchor_grid


    def initUI(self):
        # 界面初始化部分...
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片检测功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        # 添加组件到布局上
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')

    '''
       ***上传图片***
       '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    '''
    ***检测图片***
    '''

    def detect_img(self):
        img0 = cv2.imread(self.img2predict)
        det_boxes, scores, ids = infer_img(img0, self.net, self.model_h, self.model_w, self.nl, self.na, self.stride,
                                           self.anchor_grid, thred_nms=0.4, thred_cond=0.5)
        for box, score, id in zip(det_boxes, scores, ids):
            id = id.astype(int)

            label = '%s:%.2f' % (self.dic_labels[id], score)
            plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)

        resize_scale = self.output_size / img0.shape[0]
        im0 = cv2.resize(img0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    # 视频检测，逻辑基本一致，有两个功能，分别是检测摄像头的功能和检测视频文件的功能，先做检测摄像头的功能。

    '''
    ### 界面关闭事件 ### 
    '''

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "你确定要退出 ?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            import os
            os._exit(1)
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    # 视频和摄像头的主函数是一样的，不过是传入的source不同罢了
    def detect_vid(self):

        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)

        if webcam:
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(source)
        flag_det = True
        while True:
            success, img0 = cap.read()
            if success:
                # if flag_det:
                t1 = time.time()
                det_boxes, scores, ids = infer_img(img0, self.net, self.model_h, self.model_w, self.nl, self.na,
                                                   self.stride, self.anchor_grid, thred_nms=0.4, thred_cond=0.5)
                t2 = time.time()

                for box, score, id in zip(det_boxes, scores, ids):
                    id = id.astype(int)

                    label = '%s:%.2f' % (self.dic_labels[id], score)

                    plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)

                    str_FPS = "FPS: %.2f" % (1. / (t2 - t1))

                    cv2.putText(img0, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                # cv2.imshow("video", img0)
                frame_resized = cv2.resize(img0, (550, 550))
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))

                if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                    self.stopEvent.clear()
                    self.webcam_detection_btn.setEnabled(True)
                    self.mp4_detection_btn.setEnabled(True)
                    self.reset_vid()
                    break
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):

                break
            elif key & 0xFF == ord('s'):
                flag_det = not flag_det
                print(flag_det)

        cap.release()

    '''
    ### 界面重置事件 ### 
    '''

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.vid_source = '0'
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
