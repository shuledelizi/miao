#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:53:59 2019

@author: csl
"""
import matplotlib.pyplot as plt
import numpy as np
import math

n = 20
N = 72 * n   #像素点个数为（72*n）^2
pi = 3.1415
M = 64     #圆边上取点的个数/2，基于极坐标
k = 0
mm = 0
angle = np.arange(0,2*pi,pi/M) #取点所需的角度
flag_1 = False
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})#生成figure与axis
w, h = N, N
Matrix = [[0 for x in range(w)] for y in range(h)] #判断每个像素是否被填充的矩阵，默认为0
Matrix = np.array(Matrix)  #不加就报错
while k < 200:#尝试生成的图形数
    x = np.random.randint(N/32,N-N/32) #圆形的横坐标
    y = np.random.randint(N/32,N-N/32) #圆形的纵坐标
    rr = np.random.randint(N/128,N/32) #半径
    T = 2 * M #圆边上取点的个数
    xx = np.arange(T-1)
    yy = np.arange(T-1)
    for i in range(T-1): #给每个点的坐标赋值
        xx[i] = x + rr*math.cos(angle[i])
        yy[i] = y + rr*math.sin(angle[i])
    mm = mm + 1
    for p in range(x-rr,x+rr)  :#获得围成该图形的最小矩形的“是否填充”的矩阵
        for q in range(y-rr,y+rr):
            if(Matrix[p,q] == 1):#如果有已被填充的，跳出p、q的循环，也不填充
                flag_1 = True
                for m in range(x-rr,p):
                    for n in range(y-rr,q):
                        Matrix[m,n] = Matrix[m,n] - 1
                break
            else:
                flag_1 = False #如果没被填充，则在填充矩阵里记录
                Matrix[p,q] = Matrix[p,q] + 1
        if flag_1:
            break
 
    if  flag_1 == False:
        ax.fill(xx,yy) #如果整个图形都未被填充，则描绘它 
        k = k + 1
        print("尝试次数为%d"%mm)
        print("生成次数为%d"%k)
ax.set_xlim(0, N)
ax.set_ylim(0, N)
plt.show()


    
