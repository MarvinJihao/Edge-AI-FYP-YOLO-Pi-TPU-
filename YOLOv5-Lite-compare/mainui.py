# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"")
        MainWindow.resize(1081, 798)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"background-color: #FFFFFF;\n"
"font-size: 14px;")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.show_img = QLabel(self.centralwidget)
        self.show_img.setObjectName(u"show_img")
        self.show_img.setGeometry(QRect(130, 210, 381, 381))
        self.show_img.setAlignment(Qt.AlignCenter)
        self.show_img.setStyleSheet(u"QLabel {\n"
"                border: 2px solid #CCCCCC;  /* \u8fb9\u6846\u989c\u8272\u548c\u5bbd\u5ea6 */\n"
"                border-radius: 8px;  /* \u8fb9\u6846\u5706\u89d2\u534a\u5f84 */\n"
"                background-color: white;  /* \u80cc\u666f\u989c\u8272 */\n"
"                padding: 4px;  /* \u8fb9\u6846\u5185\u8fb9\u8ddd */\n"
"            }")
        self.show_img.setPixmap(QPixmap(u"source/upload.png"))
        self.b_import_img = QPushButton(self.centralwidget)
        self.b_import_img.setObjectName(u"b_import_img")
        self.b_import_img.setGeometry(QRect(160, 690, 141, 51))
        self.b_import_img.setStyleSheet(u" QPushButton {\n"
"                background-color: #008CBA;\n"
"                color: white;\n"
"                border: none;\n"
"                border-radius: 4px;\n"
"                padding: 8px 16px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: #007B9F;\n"
"            }\n"
"            QPushButton:pressed {\n"
"                background-color: #006B85;\n"
"            }")
        self.b_import_video = QPushButton(self.centralwidget)
        self.b_import_video.setObjectName(u"b_import_video")
        self.b_import_video.setGeometry(QRect(360, 690, 141, 51))
        self.b_import_video.setStyleSheet(u" QPushButton {\n"
"                background-color: #f44336;\n"
"                color: white;\n"
"                border: none;\n"
"                border-radius: 4px;\n"
"                padding: 8px 16px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: #d32f2f;\n"
"            }\n"
"            QPushButton:pressed {\n"
"                background-color: #b71c1c;\n"
"            }")
        self.b_real_seg = QPushButton(self.centralwidget)
        self.b_real_seg.setObjectName(u"b_real_seg")
        self.b_real_seg.setGeometry(QRect(570, 690, 141, 51))
        self.b_real_seg.setStyleSheet(u"   QPushButton {\n"
"                background-color: #FF9800;\n"
"                color: white;\n"
"                border: none;\n"
"                border-radius: 4px;\n"
"                padding: 8px 16px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: #F57C00;\n"
"            }\n"
"            QPushButton:pressed {\n"
"                background-color: #E65100;\n"
"            }")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(630, 230, 71, 16)
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(630, 330, 71, 16)
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(630, 550, 71, 20)
        self.polyp_area = QLabel(self.centralwidget)
        self.polyp_area.setObjectName(u"polyp_area")
        self.polyp_area.setGeometry(730, 550, 241, 20)
        self.polyp_area.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
        self.b_seg = QPushButton(self.centralwidget)
        self.b_seg.setObjectName(u"b_seg")
        self.b_seg.setGeometry(QRect(770, 690, 141, 51))
        self.b_seg.setStyleSheet(u"  QPushButton {\n"
"                background-color: #4CAF50;\n"
"                color: white;\n"
"                border: none;\n"
"                border-radius: 4px;\n"
"                padding: 8px 16px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: #45a049;\n"
"            }\n"
"            QPushButton:pressed {\n"
"                background-color: #3e8e41;\n"
"            }")
        self.logo = QLabel(self.centralwidget)
        self.logo.setObjectName(u"logo")
        self.logo.setGeometry(QRect(10, 10, 581, 121))
        self.logo.setPixmap(QPixmap(u"source/logo.png"))
        self.detect_result = QTextEdit(self.centralwidget)
        self.detect_result.setObjectName(u"detect_result")
        self.detect_result.setGeometry(QRect(730, 220, 251, 51))
        self.doctor_advice = QTextEdit(self.centralwidget)
        self.doctor_advice.setObjectName(u"doctor_advice")
        self.doctor_advice.setGeometry(QRect(730, 320, 251, 200))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"肠道息肉智能诊断系统", None))
        self.show_img.setText("")
        self.b_import_img.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u80a0\u955c\u5f71\u50cf", None))
        self.b_import_video.setText(QCoreApplication.translate("MainWindow", u"\u5bfc\u5165\u80a0\u955c\u89c6\u9891", None))
        self.b_real_seg.setText(QCoreApplication.translate("MainWindow", u"\u5b9e\u65f6\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u7ed3\u679c:", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u533b\u751f\u5efa\u8bae:", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u606f\u8089\u9762\u79ef:", None))
        self.polyp_area.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.b_seg.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b", None))
        self.logo.setText("")
        self.detect_result.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:14px; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt;\">\u65e0</span></p></body></html>", None))
        self.doctor_advice.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:14px; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt;\">\u6682\u65e0\u5efa\u8bae</span></p></body></html>", None))
    # retranslateUi
      # Update the stylesheet for the labels
        self.label.setStyleSheet('''
            QLabel {
                color: #555555;
                font-size: 16px;
                font-weight: bold;
            }
        ''')

        self.label_3.setStyleSheet('''
            QLabel {
                color: #555555;
                font-size: 16px;
                font-weight: bold;
            }
        ''')

        self.label_5.setStyleSheet('''
            QLabel {
                color: #555555;
                font-size: 16px;
                font-weight: bold;
            }
        ''')

        self.polyp_area.setStyleSheet('''
            QLabel {
                color: #333333;
                font-size: 16px;
            }
        ''')
