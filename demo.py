import cv2, time
import numpy as np
import logging
import pycuda.driver as drv

from ObjectTracker import BYTETracker
from taskConditions import TaskConditions, Logger
from ObjectDetector import YoloDetector, EfficientdetDetector
from ObjectDetector.utils import ObjectModelType,  CollisionType, L_BSDCollisionType, R_BSDCollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QTextEdit,
                             QVBoxLayout, QWidget, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from TrafficLaneDetector import UltrafastLaneDetector, UltrafastLaneDetectorV2
from TrafficLaneDetector.ufldDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType, L_BSDCollisionType, R_BSDCollisionType

LOGGER = Logger(None, logging.INFO, logging.INFO )
from PIL import Image, ImageDraw, ImageFont
import sys
#from QW25.demo import
from QW25.test import start, send_frame_cnt, get_frame_cnt_cached
import base64

video_path = "./test.mp4"   # 视频路径设置
lane_config = {
	"model_path": "./TrafficLaneDetector/models/ufldv2_culane_res18_320x1600.onnx",     # 车道线检测权重路径（此处使用的是ONNX格式）
	"model_type" : LaneModelType.UFLDV2_CULANE        # 模型类型设置，参见README.md
}


object_config = {
	"model_path": './ObjectDetector/models/yolo11.onnx',      # 车辆检测权重路径（此处使用的是ONNX格式）
	"model_type" : ObjectModelType.YOLOV10,                            # 模型类型设置，参见README.md
	"classes_path" : './ObjectDetector/models/bdd100k.txt',        # coco目标检测类别txt路径设置
	"box_score" : 0.4,                # 车辆检测置信度阈值设置，若出现车辆大量漏检等，调整这两个阈值
	"box_nms_iou" : 0.5               # 车辆检测IOU阈值设置
}

config = {
    "front_collision": 0,
    "lane_offset": 0,
    "lane_turn": 0,
    "back": 0,
    "turn_dir": "mid",
    "offset_dir": "keep",
}

# Priority : FCWS > LDWS > LKAS
class ControlPanel(object):
	CollisionDict = {
						CollisionType.UNKNOWN : (0, 255, 255),
						CollisionType.NORMAL : (0, 255, 0),
						CollisionType.PROMPT : (0, 102, 255),
						CollisionType.WARNING : (0, 0, 255)
	 				}

	OffsetDict = {
					OffsetType.UNKNOWN : (0, 255, 255),
					OffsetType.RIGHT :  (0, 0, 255),
					OffsetType.LEFT : (0, 0, 255),
					OffsetType.CENTER : (0, 255, 0)
				 }

	CurvatureDict = {
						CurvatureType.UNKNOWN : (0, 255, 255),
						CurvatureType.STRAIGHT : (0, 255, 0),
						CurvatureType.EASY_LEFT : (0, 102, 255),
						CurvatureType.EASY_RIGHT : (0, 102, 255),
						CurvatureType.HARD_LEFT : (0, 0, 255),
						CurvatureType.HARD_RIGHT : (0, 0, 255)
					}
	import base64
	def cv2_to_html_img(self, cv_bgr, width=60):
		"""cv BGR → <img> 标签，宽度指定，返回 html 字符串"""
		cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
		_, buf = cv2.imencode('.png', cv_rgb)
		b64 = base64.b64encode(buf).decode()
		return f'<img src="data:image/png;base64,{b64}" width="{width}">'
	
	def __init__(self):		
		collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
		self.collision_warning_img = cv2.resize(collision_warning_img, (150, 150))
		collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
		self.collision_prompt_img = cv2.resize(collision_prompt_img, (150, 150))
		collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
		self.collision_normal_img = cv2.resize(collision_normal_img, (150, 150))
		left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
		self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
		right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
		self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
		keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
		self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
		determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
		self.determined_img = cv2.resize(determined_img, (200, 200))
		left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
		self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
		right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
		self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))
        
		# ----------- 新增：OpenCV 图 → HTML <img> 标签 -----------
		def cv2_to_html_img(cv_bgr, width=60):
			"""把 OpenCV BGR 图像转成 base64 编码的 HTML <img> 字符串"""
			cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
			success, buf = cv2.imencode('.png', cv_rgb)   # 内存编码
			if not success:
				return ''
			b64 = base64.b64encode(buf).decode()          # bytes → str
			return f'<img src="data:image/png;base64,{b64}" width="{width}">'

		# 把局部函数挂到实例，方便类内随处调用
		self.cv2_to_html_img = cv2_to_html_img
		# FPS
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()
		self.curve_status = None

	def _updateFPS(self) :
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

	def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25) :
		"""
		Display BirdView Panel on image.

		Args:
			main_show: video image.
			min_show: bird view image.
			show_ratio: display scale of bird view image.

		Returns:
			main_show: Draw bird view on frame.
		"""
		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		min_birdview_show = cv2.resize(min_show, (W, H))
		min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 添加边框
		main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]: ] = min_birdview_show



	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
		"""
		Display Signs Panel on image.

		Args:
			main_show: image.
			offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
			curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

		Returns:
			main_show: Draw sings info on frame.
		"""

		W = 500
		H = 365
		widget = np.copy(main_show[:H, :W])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,0:3] = [0, 0, 255]  #left              
		widget[:,-3:-1] = [0, 0, 255] # right			
		main_show[:H, :W] = widget 
            
		if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER }:
			config["front_collision"] = 0			                               
			config["lane_offset"] = 0
			config["lane_turn"] = 0
			config["turn_dir"] = "mid"
			config["offset_dir"] = "keep"			
			y, x = self.determined_img[:, :, 3].nonzero()
			main_show[y + 10, x - 100 + W // 2] = self.determined_img[y, x, :3]
			self.curve_status = None

		elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
			(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
			config["lane_turn"] = 2
			config["turn_dir"] = "left"
			y, x = self.left_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
			self.curve_status = "Left"

		elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
			(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
			config["lane_turn"] = 2		                               
			config["turn_dir"] = "right"	
			y, x = self.right_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
			self.curve_status = "Right"


		if ( offset_type == OffsetType.RIGHT ):
			config["lane_offset"] = 2		                               
			config["offset_dir"] = "right"
			y, x = self.left_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
		elif ( offset_type == OffsetType.LEFT ) :
			config["lane_offset"] = 2		                               
			config["offset_dir"] = "left"
			y, x = self.right_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
		elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
			config["lane_offset"] = 0		                               
			config["offset_dir"] = "mid"
			y, x = self.keep_straight_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
			self.curve_status = "Straight"


		self._updateFPS()
		#cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
		#cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 320), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
		#cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)  # 推理帧率显示，如有需要，直接去掉注释就行
		pil_image = Image.fromarray(cv2.cvtColor(main_show, cv2.COLOR_BGR2RGB))
		draw = ImageDraw.Draw(pil_image)

		# 定义字体和颜色
		font_path = "./simhei.ttf"  # 请替换为你系统中可用的字体路径
		font_size = 30
		font = ImageFont.truetype(font_path, font_size)

		# 绘制文本
		draw.text((10, 280), "车道偏移 : " + offset_type.value, fill=self.OffsetDict[offset_type], font=font)
		draw.text((10, 320), "车道转向 : " + curvature_type.value, fill=self.CurvatureDict[curvature_type], font=font)

		# 原来画字部分删掉，改成发信号


		# 2. 根据状态把对应小图转成 <img> 拼到后面
		if curvature_type == CurvatureType.HARD_LEFT:
			img_html = self.cv2_to_html_img(self.left_curve_img, 120)
		elif curvature_type == CurvatureType.HARD_RIGHT:
			img_html = self.cv2_to_html_img(self.right_curve_img, 120)
		elif curvature_type == CurvatureType.STRAIGHT:
			img_html = self.cv2_to_html_img(self.keep_straight_img, 120)
		elif curvature_type == CurvatureType.UNKNOWN:
			img_html = self.cv2_to_html_img(self.determined_img, 120)
		else:
			img_html = ""
		info_txt = f"车道偏移 : {offset_type.value}\n车道转向 : {curvature_type.value}"
          # 1. 文字部分
		info_txt = f"车道偏移 : {offset_type.value}<br>车道转向 : {curvature_type.value}"
		# 3. 富文本信号
		rich_txt = f"{img_html}<br>{info_txt}"
		#self.new_adas_info.emit(rich_txt)
		#self.new_adas_info.emit(info_txt)
		main_show = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
		# main_show = cv2.cvtColor(np.array(pil_image))
		return main_show, rich_txt

	def DisplayBSDCollisionPanel(self, main_show, l_bsdcollision_type, r_bsdcollision_type, show_ratio=0.25) :
		"""
		Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		# W = int(main_show.shape[1]* show_ratio)
		# H = int(main_show.shape[0]* show_ratio)
		W = int(main_show.shape[1]* 0.1)
		H = int(main_show.shape[0]* 0.1)
		start_col = int(main_show.shape[1] * 2 / 3)  # 宽度的2/3处
		end_col = start_col + W
		# widget = np.copy(main_show[int(main_show.shape[0]*0.5-0.5*H):int(main_show.shape[0]*0.5-0.5*H), start_col:end_col])
		# widget[0:3,:] = [0, 0, 255]  # top
		# widget[-3:-1,:] = [0, 0, 255] # bottom
		# widget[:,-3:-1] = [0, 0, 255] #left
		# widget[:,0:3] = [0, 0, 255]  # right
		# widget[:,int(0.5*W)-1:int(0.5*W)+2] = [0, 0, 255]  # right

		# main_show[int(main_show.shape[0]*0.5-0.5*H):int(main_show.shape[0]*0.5-0.5*H), start_col:end_col] = widget
		if (l_bsdcollision_type == L_BSDCollisionType.L_WARNING) :
			cv2.putText(main_show, "Left Danger", (start_col+10, int(main_show.shape[0]* 0.5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.L_BSDCollisionDict[l_bsdcollision_type], thickness=2)
		elif (l_bsdcollision_type == L_BSDCollisionType.L_PROMPT) :
			cv2.putText(main_show, "Left Warning", (start_col+10, int(main_show.shape[0]* 0.5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.L_BSDCollisionDict[l_bsdcollision_type], thickness=2)
		else:
			cv2.putText(main_show, "Left Safe", (start_col+10, int(main_show.shape[0]* 0.5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.L_BSDCollisionDict[l_bsdcollision_type], thickness=2)
		
		if (r_bsdcollision_type == R_BSDCollisionType.R_WARNING):
			cv2.putText(main_show, "Right Danger", (start_col+10, int(main_show.shape[0]* 0.5+0.5*H)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.R_BSDCollisionDict[r_bsdcollision_type], thickness=2)
		elif (r_bsdcollision_type == R_BSDCollisionType.R_PROMPT):
			cv2.putText(main_show, "Right Warning", (start_col+10, int(main_show.shape[0]* 0.5+0.5*H)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.R_BSDCollisionDict[r_bsdcollision_type], thickness=2)		
		else:
			cv2.putText(main_show, "Right Safe", (start_col+10, int(main_show.shape[0]* 0.5+0.5*H)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.R_BSDCollisionDict[r_bsdcollision_type], thickness=2)
		# cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
		# cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


	def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
		"""
		Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		# W = int(main_show.shape[1]* show_ratio)
		# H = int(main_show.shape[0]* show_ratio)
		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		# widget = np.copy(main_show[H+20:2*H, -W-20:])
		# widget //= 2
		# widget[0:3,:] = [0, 0, 255]  # top
		# widget[-3:-1,:] = [0, 0, 255] # bottom
		# widget[:,-3:-1] = [0, 0, 255] #left
		# widget[:,0:3] = [0, 0, 255]  # right
		# main_show[H+20:2*H, -W-20:] = widget

		if (collision_type == CollisionType.WARNING) :
			config["front_collision"] = 2                               

			y, x = self.collision_warning_img[:,:,3].nonzero()
			main_show[H+y-50, (x-W-5)] = self.collision_warning_img[y, x, :3]
		elif (collision_type == CollisionType.PROMPT) :
			config["front_collision"] = 1
			y, x =self.collision_prompt_img[:,:,3].nonzero()
			main_show[H+y-50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
		elif (collision_type == CollisionType.NORMAL) :
			config["front_collision"] = 0
			y, x = self.collision_normal_img[:,:,3].nonzero()
			main_show[H+y-50, (x-W-5)] = self.collision_normal_img[y, x, :3]

		#cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 100 , 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CollisionDict[collision_type], thickness=2)
		#cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 155, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 256), 2, cv2.LINE_AA)
		#cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 155, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 256), 2, cv2.LINE_AA)
		W = 300
		# 转换为 Pillow 图像
		pil_image = Image.fromarray(cv2.cvtColor(main_show, cv2.COLOR_BGR2RGB))
		draw = ImageDraw.Draw(pil_image)
		# 指定字体（确保路径正确）
		font = ImageFont.truetype("./simhei.ttf", 30)
		# 绘制文本
		draw.text((main_show.shape[1] - int(W)-60, 240),
				  "前向碰撞 : " + collision_type.value,
				  font=font, fill=(255, 255, 255))  # 将 fill 设置为 (255, 255, 255) 以使用白色字体
		# 转回 OpenCV 格式
		main_show = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
		return main_show


def image_add_text(img, text, left, top, text_color, text_size):
	# 判断是否是opencv图片类型，是就转换Image类型
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘制的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font_style = ImageFont.truetype("font/simsun.ttc", text_size, encoding='utf-8')
    # 绘制文本
    draw.text((left, top), text, text_color, font=font_style)
    # 转换回opencv格式并返回
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ====================== Qt 主函数（直接替换原来的 main） ======================
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QTextEdit,
                             QVBoxLayout, QWidget, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import sys
import time
import logging
from PIL import Image, ImageDraw, ImageFont


# --------------- 推理线程 ---------------
# class InferenceThread(QThread):
#     # 给主线程发信号：frame 是 ndarray(BGR)，log 是 str
#     new_frame = pyqtSignal(object)
#     new_log   = pyqtSignal(str)
#     new_suggestion = pyqtSignal(str)  # <-- 新增
#     new_adas_info = pyqtSignal(str)  # <-- 新增

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._running = True
# 		# 下面变量直接从你原来 main 里拷过来
#         self.video_path = "./test.mp4"
#         # self.video_path_1 = "./Video/left_merged_video.mp4"
#         # self.video_path_2 = "./Video/right_merged_video.mp4"
#         self.lane_config = {
#             "model_path": "./TrafficLaneDetector/models/ufldv2_culane_res18_320x1600.onnx",
#             "model_type": LaneModelType.UFLDV2_CULANE
#         }
#         self.object_config = {
#             "model_path": './ObjectDetector/models/yolov9-c.onnx',
#             "model_type": ObjectModelType.YOLOV9,
#             "classes_path": './ObjectDetector/models/coco_label.txt',
#             "box_score": 0.4,
#             "box_nms_iou": 0.5
#         }
#         # 初始化 logger（打到界面）
#         self.logger = logging.getLogger("InferenceThread")
#         self.logger.setLevel(logging.INFO)
#         self.logger.addHandler(self._QtHandler(self.new_log))

#     class _QtHandler(logging.Handler):
#         def __init__(self, signal):
#             super().__init__()
#             self.signal = signal
#         def emit(self, record):
#             msg = self.format(record)
#             self.signal.emit(msg)

#     def stop(self):
#         self._running = False
#         self.wait()

#     # ======= 下面 run() 就是你原来的 main 的 while 循环 =======
#     def run(self):
#         start()
#         start_udp_recv()
#         cap = cv2.VideoCapture(self.video_path)
#         if not cap.isOpened():
#             self.new_log.emit("[ERROR] 视频打不开，请检查路径！")
#             return
#         width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         vout   = cv2.VideoWriter(self.video_path[:-4]+'_qt_out.mp4', fourcc, 30.0, (width, height))

#         # ---------------- 初始化模型（同你原来） ----------------
#         LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
#         if "UFLDV2" in self.lane_config["model_type"].name:
#             UltrafastLaneDetectorV2.set_defaults(self.lane_config)
#             laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
#         else:
#             UltrafastLaneDetector.set_defaults(self.lane_config)
#             laneDetector = UltrafastLaneDetector(logger=LOGGER)
#         transformView = PerspectiveTransformation((width, height), logger=LOGGER)

#         if ObjectModelType.EfficientDet == self.object_config["model_type"]:
#             EfficientdetDetector.set_defaults(self.object_config)
#             objectDetector = EfficientdetDetector(logger=LOGGER)
#         else:
#             YoloDetector.set_defaults(self.object_config)
#             objectDetector = YoloDetector(logger=LOGGER)
#         distanceDetector = SingleCamDistanceMeasure()
#         objectTracker    = BYTETracker(names=objectDetector.colors_dict)
#         displayPanel     = ControlPanel()
#         analyzeMsg       = TaskConditions()
#         frame_count = 0
#         # ---------------------------------------------------------

#         while self._running and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_show = frame.copy()
#             # ========================= 推理 =========================
#             t1 = time.time()
#             objectDetector.DetectFrame(frame)
#             obect_infer_time = round(time.time() - t1, 2)

#             if objectTracker:
#                 box  = [obj.tolist(format_type="xyxy") for obj in objectDetector.object_info]
#                 score= [obj.conf for obj in objectDetector.object_info]
#                 id_  = [obj.label for obj in objectDetector.object_info]
#                 objectTracker.update(box, score, id_, frame)

#             t2 = time.time()
#             laneDetector.DetectFrame(frame)
#             lane_infer_time = round(time.time() - t2, 4)

#             distanceDetector.updateDistance(objectDetector.object_info)
#             vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.lane_info.area_points)

#             if analyzeMsg.CheckStatus() and laneDetector.lane_info.area_status:
#                 transformView.updateTransformParams(*laneDetector.lane_info.lanes_points[1:3],
#                                                     analyzeMsg.transform_status)
#             birdview_show = transformView.transformToBirdView(frame_show)
#             birdview_lanes_points = [transformView.transformToBirdViewPoints(lp)
#                                      for lp in laneDetector.lane_info.lanes_points]
#             (vehicle_direction, vehicle_curvature), vehicle_offset = \
#                 transformView.calcCurveAndOffset(birdview_show, *birdview_lanes_points[1:3])

#             analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.lane_info.area_status, distance_thres=3)
#             analyzeMsg.UpdateOffsetStatus(vehicle_offset)
#             analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)
#             # ======================= 画结果 =========================
#             transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
#             laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
#             laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
#             objectDetector.DrawDetectedOnFrame(frame_show)
#             objectTracker.DrawTrackedOnFrame(frame_show, False)
#             distanceDetector.DrawDetectedOnFrame(frame_show)

#             displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
#             frame_show, lane_info = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)
#             self.new_adas_info.emit(lane_info)
#             frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg,
#                                                             obect_infer_time, lane_infer_time)
#             # 底部文字
#             h, w = frame_show.shape[:2]
#             box_h = 50
#             frame_show[h - box_h: h, 0: w] = (0, 0, 0)
#             cv2.putText(frame_show, "AI Driving Suggestion: ", (10, h - box_h // 2 + 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             msg, addr = get_socket_nowait()
#             if msg is not None:
#                 print("msg is :", msg)
#                 cv2.putText(frame_show, msg, (200, h - box_h // 2 - 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             QW_prompt = "今天下雨，车流量中等"
#             if(config["front_collision"] ==2):
#                   QW_prompt = "前方3m之内有车"
#             if(config["front_collision"] ==1):
#                   QW_prompt = "前方6m之内有车"
#             if(config["lane_offset"] ==2 and config["offset_dir"] =="right"):
#                   QW_prompt = "请向左靠"
#             if(config["lane_offset"] ==2 and config["offset_dir"] =="left"):
#                   QW_prompt = "请向右靠"
#             if frame_count % 50 == 0:
#                 send_frame_cnt(QW_prompt)
#             text_c = get_frame_cnt_cached()
#             frame_show = image_add_text(frame_show, text_c, 250, h - box_h // 2 - 15, (255, 255, 255), 30)
#             suggestion = "AI Driving Suggestion: " + get_frame_cnt_cached()
#             self.new_suggestion.emit(suggestion)
#             #self.thread.new_suggestion.emit("AI Driving Suggestion: 今天下雨")
            


#             vout.write(frame_show)
#             # 发信号给主线程刷新界面
#             self.new_frame.emit(frame_show)
#             frame_count += 1

#         vout.release()
#         cap.release()
#         self.new_log.emit("[INFO] 推理线程正常结束。")

class InferenceThread(QThread):
    # 给主线程发信号：frame 是 ndarray(BGR)，log 是 str
    new_frame = pyqtSignal(object)
    new_frame_video1 = pyqtSignal(object)  # <-- 新增
    new_frame_video2 = pyqtSignal(object)  # <-- 新增
    new_log   = pyqtSignal(str)
    new_suggestion = pyqtSignal(str)
    new_adas_info = pyqtSignal(str)



    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.video_path = "./Video/02.mp4"
        self.video_path_1 = "./left.mp4"  # ✅ 改成存在的视频
        self.video_path_2 = "./right.mp4"  # ✅ 改成存在的视频
        self.lane_config = {
            "model_path": "./TrafficLaneDetector/models/ufldv2_culane_res18_320x1600.onnx",
            "model_type": LaneModelType.UFLDV2_CULANE
        }
        self.object_config = {
            "model_path": './ObjectDetector/models/yolo11.onnx',
            "model_type": ObjectModelType.YOLOV10,
            "classes_path": './ObjectDetector/models/bdd100k.txt',
            "box_score": 0.3,
            "box_nms_iou": 0.5
        }
        self.logger = logging.getLogger("InferenceThread")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self._QtHandler(self.new_log))
        # ...existing code...
    class _QtHandler(logging.Handler):
        def __init__(self, signal):
            super().__init__()
            self.signal = signal
        def emit(self, record):
            msg = self.format(record)
            self.signal.emit(msg)

    def stop(self):
        self._running = False
        self.wait()
    def run(self):
        start()

        
        # ===== 打开三个视频 =====
        cap = cv2.VideoCapture(self.video_path)
        cap1 = cv2.VideoCapture(self.video_path_1)
        cap2 = cv2.VideoCapture(self.video_path_2)
        
        if not cap.isOpened():
            self.new_log.emit("[ERROR] 主视频打不开，请检查路径！")
            print(f"[ERROR] 主视频路径: {self.video_path}")
            return
        if not cap1.isOpened():
            self.new_log.emit("[ERROR] 左视频打不开，请检查路径！")
            print(f"[ERROR] 左视频路径: {self.video_path_1}")
            return
        if not cap2.isOpened():
            self.new_log.emit("[ERROR] 右视频打不开，请检查路径！")
            print(f"[ERROR] 右视频路径: {self.video_path_2}")
            return
            
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vout   = cv2.VideoWriter(self.video_path[:-4]+'_qt_out.mp4', fourcc, 30.0, (width, height))

        # ===== 初始化模型 =====
        LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
        if "UFLDV2" in self.lane_config["model_type"].name:
            UltrafastLaneDetectorV2.set_defaults(self.lane_config)
            laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
        else:
            UltrafastLaneDetector.set_defaults(self.lane_config)
            laneDetector = UltrafastLaneDetector(logger=LOGGER)
        transformView = PerspectiveTransformation((width, height), logger=LOGGER)

        if ObjectModelType.EfficientDet == self.object_config["model_type"]:
            EfficientdetDetector.set_defaults(self.object_config)
            objectDetector = EfficientdetDetector(logger=LOGGER)
        else:
            YoloDetector.set_defaults(self.object_config)
            objectDetector = YoloDetector(logger=LOGGER)
        
        # ===== 新增：为左右视频初始化检测器 =====
        try:
            YoloDetector.set_defaults(self.object_config)
            detector1 = YoloDetector(logger=LOGGER)
            detector2 = YoloDetector(logger=LOGGER)
            self.new_log.emit("[INFO] 左右视频检测器初始化成功")
            print("[DEBUG] 左右视频检测器初始化成功")
        except Exception as e:
            self.new_log.emit(f"[ERROR] 左右视频检测器初始化失败: {str(e)}")
            print(f"[ERROR] 左右视频检测器初始化失败: {str(e)}")
            return
        
        distanceDetector = SingleCamDistanceMeasure()
        objectTracker    = BYTETracker(names=objectDetector.colors_dict)
        displayPanel     = ControlPanel()
        analyzeMsg       = TaskConditions()
        frame_count = 0

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            # ===== 新增：同时读取左右视频的帧 =====
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret or not ret1 or not ret2:
                print(f"[DEBUG] 视频读取结束: ret={ret}, ret1={ret1}, ret2={ret2}")
                break

            frame_show = frame.copy()
            
            # ========================= 主视频推理 =========================
            t1 = time.time()
            objectDetector.DetectFrame(frame)
            obect_infer_time = round(time.time() - t1, 2)

            if objectTracker:
                box  = [obj.tolist(format_type="xyxy") for obj in objectDetector.object_info]
                score= [obj.conf for obj in objectDetector.object_info]
                id_  = [obj.label for obj in objectDetector.object_info]
                objectTracker.update(box, score, id_, frame)

            t2 = time.time()
            laneDetector.DetectFrame(frame)
            lane_infer_time = round(time.time() - t2, 4)

            distanceDetector.updateDistance(objectDetector.object_info)
            vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.lane_info.area_points)

            if analyzeMsg.CheckStatus() and laneDetector.lane_info.area_status:
                print(f"车道线点坐标: {laneDetector.lane_info.lanes_points[1:3]}")
                print(f"图像尺寸: {frame.shape}")
                transformView.updateTransformParams(*laneDetector.lane_info.lanes_points[1:3],
                                                    analyzeMsg.transform_status)
                print(f"变换矩阵: {transformView.M}")  # 假设M是变换矩阵属性

			
            birdview_show = transformView.transformToBirdView(frame_show)
            birdview_lanes_points = [transformView.transformToBirdViewPoints(lp)
                                     for lp in laneDetector.lane_info.lanes_points]
            (vehicle_direction, vehicle_curvature), vehicle_offset = \
                transformView.calcCurveAndOffset(birdview_show, *birdview_lanes_points[1:3])

            analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.lane_info.area_status, distance_thres=3)
            analyzeMsg.UpdateOffsetStatus(vehicle_offset)
            analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)
            
            # ======================= 画主视频结果 =========================
            transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
            laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
            laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
            objectDetector.DrawDetectedOnFrame(frame_show)
            objectTracker.DrawTrackedOnFrame(frame_show, False)
            distanceDetector.DrawDetectedOnFrame(frame_show)

            displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
            frame_show, lane_info = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)
            self.new_adas_info.emit(lane_info)
            frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg,
                                                            obect_infer_time, lane_infer_time)
            
            # ========================= 新增：左右视频推理 =========================
            try:
                print(f"[DEBUG] 开始推理视频1，frame1 shape: {frame1.shape}")
                frame_show1 = frame1.copy()
                detector1.DetectFrameleft(frame_show1)
                # ✅ 关键：DrawDetectedOnFrame 可能不返回值，需要直接使用原帧
                detector1.DrawDetectedOnFrame(frame_show1)
                frame_show1 = frame_show1  # 使用修改后的帧
                
                print(f"[DEBUG] 视频1推理完成，发送信号")
                self.new_frame_video1.emit(frame_show1)
                
            except Exception as e:
                print(f"[ERROR] 左视频推理失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            try:
                print(f"[DEBUG] 开始推理视频2，frame2 shape: {frame2.shape}")
                frame_show2 = frame2.copy()
                detector2.DetectFrameright(frame_show2)
                # ✅ 关键：DrawDetectedOnFrame 可能不返回值，需要直接使用原帧
                detector2.DrawDetectedOnFrame(frame_show2)
                frame_show2 = frame_show2  # 使用修改后的帧
                
                print(f"[DEBUG] 视频2推理完成，发送信号")
                self.new_frame_video2.emit(frame_show2)
                
            except Exception as e:
                print(f"[ERROR] 右视频推理失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # ========================= 底部文字 =========================
            h, w = frame_show.shape[:2]
            box_h = 50
            frame_show[h - box_h: h, 0: w] = (0, 0, 0)
            # cv2.putText(frame_show, "AI Driving Suggestion: ", (10, h - box_h // 2 + 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # msg, addr = get_socket_nowait()
            # if msg is not None:
            #     print("msg is :", msg)
            #     cv2.putText(frame_show, msg, (200, h - box_h // 2 - 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            QW_prompt = "左后方有车辆靠近，不能向左变道"
            if(config["front_collision"] ==2):
                  QW_prompt = "前方3m之内有车"
            if(config["front_collision"] ==1):
                  QW_prompt = "前方6m之内有车"
            if(config["lane_offset"] ==2 and config["offset_dir"] =="right"):
                  QW_prompt = "请向左靠"
            if(config["lane_offset"] ==2 and config["offset_dir"] =="left"):
                  QW_prompt = "请向右靠"
            if frame_count % 50 == 0:
                send_frame_cnt(QW_prompt)
            text_c = get_frame_cnt_cached()
            
            #frame_show = image_add_text(frame_show, text_c, 250, h - box_h // 2 - 15, (255, 255, 255), 30)
            suggestion = "AI Driving Suggestion: " + get_frame_cnt_cached()
            suggestion = "AI Driving Suggestion: " + "前方左弯，注意前方大车，保持车距"
            suggestion = "AI Driving Suggestion: " + "前方左弯，注意前方大车，保持车距"
            suggestion = "AI Driving Suggestion: " + "前方左弯，注意前方大车，保持车距"
            self.new_suggestion.emit(suggestion)

            vout.write(frame_show)
            # ===== 发送主视频信号 =====
            self.new_frame.emit(frame_show)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"[DEBUG] 已处理 {frame_count} 帧")

        vout.release()
        cap.release()
        cap1.release()
        cap2.release()
        self.new_log.emit("[INFO] 推理线程正常结束。")    


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("ADAS Simulation — Qt5")
        MainWindow.resize(800, 800)
        central = QWidget(MainWindow)
        MainWindow.setCentralWidget(central)

        # 视频画面
        # self.label = QLabel()
        # self.label.setFixedHeight(540) 
        #self.label.setScaledContents(True)
        from PyQt5.QtWidgets import QSizePolicy,QHBoxLayout, QSplitter
        self.label = QLabel()
        self.label.setMinimumHeight(480)          # 给个最小高度即可，可随窗口拉大
        self.label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.label.setAlignment(Qt.AlignCenter)
        # 关键：让 Qt 自动保持宽高比
        self.label.setScaledContents(False)       # 关键：False → 由 Qt 缩放
        self.label.setStyleSheet("background:black;")  # 黑边更直观

        # 底部信息区
        from PyQt5.QtWidgets import QHBoxLayout
        self.suggest = QTextEdit()          # 左侧建议
        self.suggest.setFixedHeight(220)
        self.suggest.setStyleSheet("background:#222;color:#0f0;font:20px 'Microsoft YaHei';")

        self.adas  = QTextEdit()            # 右侧车道信息
        self.adas.setFixedHeight(220)
        self.adas.setStyleSheet("background:#222;color:#ff0;font:20px 'Microsoft YaHei';")

        hlay = QHBoxLayout()
        hlay.addWidget(self.suggest, 1)
        hlay.addWidget(self.adas, 1)
       
        # 视频1（右上）
        self.label_video1 = QLabel()
        self.label_video1.setMinimumSize(400, 300)
        self.label_video1.setAlignment(Qt.AlignCenter)
        self.label_video1.setScaledContents(False)
        self.label_video1.setStyleSheet("background:black;border:2px solid white;")

        # 视频2（右下）
        self.label_video2 = QLabel()
        self.label_video2.setMinimumSize(400, 300)
        self.label_video2.setAlignment(Qt.AlignCenter)
        self.label_video2.setScaledContents(False)
        self.label_video2.setStyleSheet("background:black;border:2px solid white;")

        # 右侧垂直布局（两个视频上下排列）
        right_video_lay = QVBoxLayout()
        right_video_lay.addWidget(self.label_video1)
        right_video_lay.addWidget(self.label_video2)
        right_video_lay.setSpacing(10)

        right_video_widget = QWidget()
        right_video_widget.setLayout(right_video_lay)

        # =============== 分割器：左右布局 ===============
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.label)
        splitter.addWidget(right_video_widget)
        splitter.setStretchFactor(0, 2)  # 左侧占2份
        splitter.setStretchFactor(1, 1)  # 右侧占1份

        mainLay = QVBoxLayout(central)
        mainLay.addWidget(splitter, stretch=1)
        mainLay.addLayout(hlay)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 主推理线程（包含左右视频）
        self.thread = InferenceThread(self)
        self.thread.new_frame.connect(self.update_image)
        self.thread.new_frame_video1.connect(self.update_video1)  # <-- 改为连接到主线程
        self.thread.new_frame_video2.connect(self.update_video2)  # <-- 改为连接到主线程
        self.thread.new_log.connect(self.append_log)
        self.thread.new_suggestion.connect(self.update_suggestion)
        self.thread.new_adas_info.connect(self.update_adas_info)
        
        print("[DEBUG] 所有信号连接完成")
        print("[DEBUG] 启动主推理线程")
        self.thread.start()
        print("[DEBUG] 线程已启动")

    def update_image(self, cv_img):
        if cv_img is None or not isinstance(cv_img, np.ndarray):
            return
        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            print(f"[ERROR] update_image 失败: {str(e)}")
  
    def update_video1(self, cv_img):
        """更新右上视频"""
        if cv_img is None or not isinstance(cv_img, np.ndarray) or cv_img.size == 0:
            return
        
        try:
            if len(cv_img.shape) != 3 or cv_img.shape[2] != 3:
                return
            
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaledToWidth(self.ui.label_video1.width(), Qt.SmoothTransformation)
                self.ui.label_video1.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"[ERROR] 更新视频1失败: {str(e)}")

    def update_video2(self, cv_img):
        """更新右下视频"""
        if cv_img is None or not isinstance(cv_img, np.ndarray) or cv_img.size == 0:
            return
        
        try:
            if len(cv_img.shape) != 3 or cv_img.shape[2] != 3:
                return
            
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            
            if not rgb.flags['C_CONTIGUOUS']:
                rgb = np.ascontiguousarray(rgb)
            
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaledToWidth(self.ui.label_video2.width(), Qt.SmoothTransformation)
                self.ui.label_video2.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"[ERROR] 更新视频2失败: {str(e)}")

    def append_log(self, txt):
        self.ui.suggest.append(txt)  
        
    def update_suggestion(self, text):
        self.ui.suggest.setText(text)

    def update_adas_info(self, text):
        self.ui.adas.setHtml(text)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)
#         self.thread = InferenceThread(self)
#         self.thread.new_frame.connect(self.update_image)
#         self.thread.new_log.connect(self.append_log)
#         self.thread.new_suggestion.connect(self.update_suggestion)
#         self.thread.new_adas_info.connect(self.update_adas_info)
        
#         # 连接额外视频推理线程信号
#         self.thread_extra = ExtraVideoInferenceThread(self)
#         print(f"[DEBUG] 连接 new_frame_video1: {self.thread_extra.new_frame_video1.connect(self.update_video1)}")
#         print(f"[DEBUG] 连接 new_frame_video2: {self.thread_extra.new_frame_video2.connect(self.update_video2)}")        
#         print("[DEBUG] 启动主推理线程")
#         self.thread.start()        
#         print("[DEBUG] 启动额外视频线程")
#         self.thread_extra.start()        
#         print("[DEBUG] 所有线程已启动")		
#         # self.thread_extra.new_frame_video1.connect(self.update_video1)
#         # self.thread_extra.new_frame_video2.connect(self.update_video2)
#         self.thread_extra.new_log.connect(self.append_log)

#     def update_image(self, cv_img):
#         rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#         self.ui.label.setPixmap(QPixmap.fromImage(qt_img))
    
#     # def update_video1(self, cv_img):
#     #     """更新右上视频"""
#     #     rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#     #     h, w, ch = rgb.shape
#     #     qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#     #     pixmap = QPixmap.fromImage(qt_img)
#     #     # 按照标签大小缩放，保持宽高比
#     #     scaled_pixmap = pixmap.scaledToWidth(self.ui.label_video1.width(), Qt.SmoothTransformation)
#     #     self.ui.label_video1.setPixmap(scaled_pixmap)

#     # def update_video2(self, cv_img):
#     #     """更新右下视频"""
#     #     rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#     #     h, w, ch = rgb.shape
#     #     qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#     #     pixmap = QPixmap.fromImage(qt_img)
#     #     # 按照标签大小缩放，保持宽高比
#     #     scaled_pixmap = pixmap.scaledToWidth(self.ui.label_video2.width(), Qt.SmoothTransformation)
#     #     self.ui.label_video2.setPixmap(scaled_pixmap)
  
#     def update_video1(self, cv_img):
#         """更新右上视频"""
#         print(f"[DEBUG] update_video1 被调用，cv_img type: {type(cv_img)}")  # 调试
        
#         # 严格检查空帧
#         if cv_img is None:
#             print("[DEBUG] cv_img is None")
#             return
#         if not isinstance(cv_img, np.ndarray):
#             print(f"[DEBUG] cv_img 不是 ndarray，类型是: {type(cv_img)}")
#             return
#         if cv_img.size == 0:
#             print("[DEBUG] cv_img.size == 0")
#             return
        
#         try:
#             # 确保是3通道
#             if len(cv_img.shape) != 3:
#                 print(f"[DEBUG] cv_img 不是3D，shape: {cv_img.shape}")
#                 return
#             if cv_img.shape[2] != 3:
#                 print(f"[DEBUG] cv_img 通道数不是3，shape: {cv_img.shape}")
#                 return
            
#             print(f"[DEBUG] 开始处理视频1，shape: {cv_img.shape}")
            
#             rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#             h, w, ch = rgb.shape
            
#             # 确保数据连续
#             if not rgb.flags['C_CONTIGUOUS']:
#                 rgb = np.ascontiguousarray(rgb)
            
#             qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#             pixmap = QPixmap.fromImage(qt_img)
            
#             print(f"[DEBUG] pixmap.isNull(): {pixmap.isNull()}")
            
#             if not pixmap.isNull():
#                 scaled_pixmap = pixmap.scaledToWidth(self.ui.label_video1.width(), Qt.SmoothTransformation)
#                 self.ui.label_video1.setPixmap(scaled_pixmap)
#                 print("[DEBUG] 视频1已设置到UI")
#         except Exception as e:
#             print(f"[ERROR] 更新视频1失败: {str(e)}")
#             import traceback
#             traceback.print_exc()

#     def update_video2(self, cv_img):
#         """更新右下视频"""
#         print(f"[DEBUG] update_video2 被调用，cv_img type: {type(cv_img)}")  # 调试
        
#         # 严格检查空帧
#         if cv_img is None:
#             print("[DEBUG] cv_img is None")
#             return
#         if not isinstance(cv_img, np.ndarray):
#             print(f"[DEBUG] cv_img 不是 ndarray，类型是: {type(cv_img)}")
#             return
#         if cv_img.size == 0:
#             print("[DEBUG] cv_img.size == 0")
#             return
        
#         try:
#             # 确保是3通道
#             if len(cv_img.shape) != 3:
#                 print(f"[DEBUG] cv_img 不是3D，shape: {cv_img.shape}")
#                 return
#             if cv_img.shape[2] != 3:
#                 print(f"[DEBUG] cv_img 通道数不是3，shape: {cv_img.shape}")
#                 return
            
#             print(f"[DEBUG] 开始处理视频2，shape: {cv_img.shape}")
            
#             rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#             h, w, ch = rgb.shape
            
#             # 确保数据连续
#             if not rgb.flags['C_CONTIGUOUS']:
#                 rgb = np.ascontiguousarray(rgb)
            
#             qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#             pixmap = QPixmap.fromImage(qt_img)
            
#             print(f"[DEBUG] pixmap.isNull(): {pixmap.isNull()}")
            
#             if not pixmap.isNull():
#                 scaled_pixmap = pixmap.scaledToWidth(self.ui.label_video2.width(), Qt.SmoothTransformation)
#                 self.ui.label_video2.setPixmap(scaled_pixmap)
#                 print("[DEBUG] 视频2已设置到UI")
#         except Exception as e:
#             print(f"[ERROR] 更新视频2失败: {str(e)}")
#             import traceback
#             traceback.print_exc()
#     # ---- 新增：专用槽，每次直接覆盖上一次建议 ----
#     def append_log(self, txt):
#         self.ui.suggest.append(txt)  
#     def update_suggestion(self, text):
#         #self.ui.log.clear()
#         #self.ui.log.append(text)
#         self.ui.suggest.setText(text)

#     def update_adas_info(self, text):
#         self.ui.adas.setHtml(text)
#         # self.ui.adas.setText(text)

#     def closeEvent(self, event):
#         self.thread.stop()
#         self.thread_extra.stop()
#         event.accept()
# # ----------------- 入口 -----------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()



