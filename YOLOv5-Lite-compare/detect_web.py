from datetime import datetime
import random

from flask import Flask, request, render_template, Response, url_for, jsonify, redirect, flash
import os
import cv2 as cv
from flask import send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


app = Flask(__name__, static_folder='static')


@app.route('/index')
def video_template():
    return render_template("index.html/")

@app.route('/upload_pic', methods=['POST'])
def uopload_pic():
    result = ""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # 获取安全的文件名 正常文件名
        filename = secure_filename(file.filename)

        # 生成随机数
        random_num = random.randint(0, 100)

        file_path = "./static/image/"  # basedir 代表获取当前位置的绝对路径

        # 如果文件夹不存在，就创建文件夹
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file.save(os.path.join(file_path, "detect.jpg"))
        result = is_Normal()
    return result


@app.route('/uopload_Android_pic', methods=['POST'])
def uopload_Android_pic():
    result = ""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # 获取安全的文件名 正常文件名
        filename = secure_filename(file.filename)

        # 生成随机数
        random_num = random.randint(0, 100)

        file_path = "./static/image/"  # basedir 代表获取当前位置的绝对路径

        # 如果文件夹不存在，就创建文件夹
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file.save(os.path.join(file_path, "detect.jpg"))
        result = is_Normal()
    return result

def is_Normal():

    return "200"


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
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

@torch.no_grad()
def model_load(weights="",  # model.pt path(s)
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               half=False,  # use FP16 half-precision inference
               dnn=False,  # use OpenCV DNN for ONNX inference
               ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model


@app.route('/detect')
def show_pic():
    return send_file('./static/image/detect_output.jpg', mimetype='image/gif')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, debug=True)