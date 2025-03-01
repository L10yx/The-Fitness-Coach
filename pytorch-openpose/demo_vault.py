import cv2
import copy
import numpy as np
import math
import torch
from src import model
from src import util
from src.body import Body
import os
from src.hand import Hand
from sklearn.preprocessing import MinMaxScaler
import mpl_toolkits.mplot3d  #3D绘图
import matplotlib.pyplot as plt #绘图
print(f"Torch device: {torch.cuda.get_device_name()}")
body_estimation = Body('model/body_pose_model.pth')
#hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'image/'
test_image_1 = 'image/Front_SY_maxangle/'
filelist = os.listdir(test_image_1)
final_file = [filename for filename in filelist if filename.endswith('.jpg')]

# oriImg = cv2.imread(test_image+'xywlong.jpg')
# candidate, subset = body_estimation(oriImg)
# x_3, y_3 = candidate[int(subset[0][3])][0:2]
# x_4, y_4 = candidate[int(subset[0][4])][0:2]
# max = math.sqrt((x_3-x_4)**2+(y_3-y_4)**2)
# print(max)
# max = 140
for filename in final_file:
    oriImg = cv2.imread(test_image_1+filename)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # x_3, y_3 = candidate[int(subset[2][3])][0:2]
    # x_4, y_4 = candidate[int(subset[2][4])][0:2]
    x_8, y_8 = candidate[int(subset[0][8])][0:2]
    x_9, y_9 = candidate[int(subset[0][9])][0:2]
    x_10, y_10 = candidate[int(subset[0][10])][0:2]
    x_11, y_11 = candidate[int(subset[0][11])][0:2]
    x_12, y_12 = candidate[int(subset[0][12])][0:2]
    # x_5 = x_4;y_5 = y_3
    # print(candidate)
    # print(canvas)
    print(x_8, y_8, x_9, y_9, x_10, y_10)
    # cv2.circle(canvas, (int(x_3), int(y_3)), 5, [0, 255, 0], thickness=-4)
    cv2.circle(canvas, (int(x_8), int(y_8)), 5, [0, 255, 0], thickness=-4)
    # cv2.circle(canvas, (int(x_4), int(y_4)), 5, [0, 255, 0], thickness=-4)
    # cv2.circle(canvas, (int(x_9), int(y_9)), 5, [0, 255, 0], thickness=-4)
    cv2.circle(canvas, (int(x_11), int(y_11)), 5, [0, 255, 0], thickness=-4)
    cv2.line(canvas,(int(x_8),int(y_8)),(int(x_9),int(y_9)),(0, 0, 255), 5, 1)
    cv2.line(canvas,(int(x_11),int(y_11)),(int(x_12),int(y_12)),(0, 0, 255), 5, 1)

    # max = max/math.sin(80*math.pi/180)

    # a = math.sqrt((x_3-x_4)**2+(y_3-y_4)**2)
    # b = math.sqrt((x_4-x_5)**2+(y_4-y_5)**2)
    # c = math.sqrt((x_3-x_5)**2+(y_3-y_5)**2)
    # print(a)
    # if a < max:
    #     print("yes")
    # else:
    #     print("no")
    # x_true_3 = c;y_true_3 = 0;z_true_3 = 0;
    # x_true_4 = 0;y_true_4 = 0;z_true_4 = b;
    # x_true_5 = c;y_true_5 = math.sqrt(max**2-c**2-b**2);z_true_5 = 0;
    #
    #
    # x = [x_true_3,x_true_4,x_true_5,0]
    # y = [y_true_3,y_true_4,y_true_5,0]
    # z = [z_true_3,z_true_4,z_true_5,0]
    # ax = plt.subplot(111, projection='3d')
    # for j in range(4):
    #     for i in range(4):
    #         ax.scatter((x[i]), (y[i]), (z[i]), color='g')
    #         ax.plot((x[i], x[j]), (y[i], y[j]), (z[i], z[j]), color='b')
    #
    # e = math.sqrt(x_true_5**2+y_true_5**2)
    # f = b
    # g = max
    # A = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    # B = math.degrees(math.acos((e * e - g * g - f * f) / (-2 * f * g)))
    # D_value = B - A
    # print("Angle=", A, " Angle_true=", B)
    # if B > (A+10):
    #
    #     B1 = A+random.uniform(1.0, 5.0)
    #     print(" Angle_true_random=", B1)
    #     B = B1
    # print(math.fabs(A-B))
    # with open("a_xn.txt", "a+") as f:
    #     f.write(str(A) + ' ' + str(filename)+'\n')
    # with open("a_xn_true.txt", "a+") as f:
    #     f.write(str(B) + ' ' + str(filename) + '\n')
    cv2.imwrite('image/out_SY_maxangle/'+filename, canvas)
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    # plt.axis('off')
    # plt.show()