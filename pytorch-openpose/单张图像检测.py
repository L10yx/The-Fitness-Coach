

#导入工具包
#导入工具包
#导入工具包


#opencv-python
import cv2

#mediapipe人工智能工具包
import mediapipe as mp

#进度条库
import time

#导入Python绘图matplotlib
import matplotlib.pyplot as plt


mp_drawing = mp.solutions.drawing_utils  #画图是必要的
mp_drawing_styles = mp.solutions.drawing_styles
#选择需要的解决方案，手部检测就mp_hands=mp.solutions.hands,其他类似
mp_holistic = mp.solutions.holistic

import math


#%matplotlib inline


#定义可视化函数
def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


#导入模型
#导入模型
#导入模型

#导入solution
mp_pose = mp.solutions.pose
#导入绘图函数
mp_drawing = mp.solutions.drawing_utils
#导入模型
pose = mp_pose.Pose(static_image_mode=True,        #是静态图片还是连续视频帧
                    model_complexity=1,            #选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    #smooth_landmarks=True,        #是否平滑关键点
                    enable_segmentation=True,      #是否人体抠图
                    min_detection_confidence=0.5,  #置信度阈值
                    min_tracking_confidence=0.5,   #追踪阈值
                    )




#读入图像
#读入图像
#读入图像

#从图片文件读入图像，opencv读入为BGR格式
img = cv2.imread('D:/E/QQ20221228185602.jpg')


#将图像输入模型，获取预测结果
#将图像输入模型，获取预测结果
#将图像输入模型，获取预测结果

#BGR转RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#将RGB图像输入模型，获取预测结果
results = pose.process(img_RGB)



#可视化检测结果
#可视化检测结果
#可视化检测结果

mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#img, results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS
look_img(img)

#在三维真实物理坐标系中可视化以米为单位的检测结果(绘制三维)
mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


#plt.show(look_img)
#plt.show(mp_drawing.plot_landmarks)




