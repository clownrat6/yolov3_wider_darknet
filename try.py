# one_record = open('data/train/0.txt','r')
# print(one_record.readlines())
import cv2
import numpy as np
from util.function import *
# a = {0: [(np.array([113.18439, 172.80754, 219.25995, 260.6206 ], dtype=np.float32), 0.67143375)]}
# temp = a[0]
# b = {}
# print(len(b))
# # print(temp[0][0])
# for box,score in temp:
#     print(box, score)
# b = np.array((1,2))
# print(np.tile(b,[2]))
# temp = cv2.imread('1.jpg')
# w,h,_ = temp.shape
# print(w,h)
# # temp = cv2.line(temp, (h/3,0), (h/3,w), (0,0,0), 5)
# temp = cv2.rectangle(temp, (100,200),(150,250), (0,0,0),5)
# cv2.imshow('sdaf',temp)
# w,h,_ = temp.shape
# length_norm = max(w,h)
# zeros = np.zeros((length_norm,length_norm,3),dtype=np.uint8)
# zeros[:w,:h,:] = temp
# cv2.imshow('zeros',zeros)
# temp = cv2.resize(zeros,(500,500))
# cv2.imshow('sda',temp)
# cv2.waitKey(0)
# from util.function import pad_image
from util.function import get_classes
print(get_classes('config/wider.names'))
# import time
# cap = cv2.VideoCapture(0)
# while(True):
#     flag,frame = cap.read()
#     w,h,_ = frame.shape
#     if not flag:
#         break
#     # length_norm = max(w,h)
#     # temp = pad_image(frame,(416,416))
#     # point = (200,300)
#     # temp = cv2.circle(temp,point,5,(255,255,255),5)
#     # point = (200*length_norm/416,300*length_norm/416)
#     # point = (int(point[0]),int(point[1]))
#     # frame = cv2.circle(frame,point,5,(255,255,255),5)
#     start = time.time()
#     cv2.imshow('cap',frame)
#     end = time.time()
#     print(start-end)
#     # cv2.imshow('cap1',temp)
#     cv2.waitKey(1)
# color = tuple(np.random.randint(0, 256, 3))
# print(type(color[0]))
# print(type(32))
# # print(
# print(type(tuple([1,2,3])[0]))
# import random
# while(True):
#     temp = random.randint(0,255)
#     if(temp > 255):
#         print(temp)
# print(get_anchors('config/wider_face_anchors.txt'))