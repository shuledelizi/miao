# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:30:03 2019

@author: 薛
"""

import matplotlib.pyplot as plt
import numpy as np
import math
pi = 3.1415
cos = math.cos
sin = math.sin
def rotation_matrix(rotation):
    R = rotation
    matrix = [[cos(R),-sin(R)],
              [sin(R), cos(R)]]
    return np.array(matrix)

def parameter_circle(x,y,s,p,rotation):
    #参数, x,y 为生成图形位置; S 为生成图形面积; p 为长宽比
    M = 64
    T = M * 2 #T为拟合的边框数
    xx = np.arange(T-1)
    yy = np.arange(T-1)#为绘制的边框的点预留内存
    angle = np.arange(0,2*pi,pi/M) #取点所需的角度
    rr = int((s / pi)**(0.5))
    for i in range(T-1): #给每个点的坐标赋值
        xx[i] = x + rr*cos(angle[i])
        yy[i] = y + rr*sin(angle[i])
    
    r1,r2 = (rr,rr)
    return xx,yy,r1,r2

def parameter_ellipse(x,y,s,p,rotation):
    #参数, x,y 为生成图形位置; S 为生成图形面积; p 为长宽比
    R = -rotation#椭圆参数方程的锅！
    M = 64
    T = M * 2 #T为拟合的边框数
    xx = np.arange(T-1)
    yy = np.arange(T-1)#为绘制的边框的点预留内存
    angle = np.arange(0,2*pi,pi/M) #取点所需的角度
    r1 = int((s/p/pi)**(0.5))
    r2 = int((s*p/pi)**(0.5))
    for i in range(T-1): #给每个点的坐标赋值
        t = angle[i]
        xx[i] = r1*cos(t)*cos(R)-r2*sin(t)*sin(R) + x
        yy[i] = r1*cos(t)*sin(R)+r2*sin(t)*cos(R) + y

    return xx,yy,r1,r2

def parameter_rectangle(x,y,s,p,rotation):
    #参数, x,y 为生成图形位置; S 为生成图形面积; p 为长宽比
    R = rotation
    xx = np.arange(4)
    yy = np.arange(4)#为绘制的边框的点预留内存

    r1 = int((s/p)**(0.5)/2)
    r2 = int((s*p)**(0.5)/2)
    tx = [-r1, r1, r1,-r1]
    ty = [ r2, r2,-r2,-r2]
    for i in range(4): #给每个点的坐标赋值
        xx[i] = np.dot(np.array([tx[i],ty[i]]).T,rotation_matrix(R))[0] + x
        yy[i] = np.dot(np.array([tx[i],ty[i]]).T,rotation_matrix(R))[1] + y

    return xx,yy,r1,r2

def parameter_triangle(x,y,s,p,rotation):
    #参数, x,y 为生成图形位置; S 为生成图形面积; p 为长宽比
    R = rotation
    xx = np.arange(3)
    yy = np.arange(3)#为绘制的边框的点预留内存
    
    r1 = int((2*s/p)**(0.5)/2)
    r2 = int((2*s*p)**(0.5)/2)
    
    if np.random.random() > 0.5:
        tx = [ 0, r1,-r1]
        ty = [r2,-r2,-r2]
    else:
        tx = [r1,-r1,-r1]
        ty = [ 0,-r2,r2]
    for i in range(3): #给每个点的坐标赋值
            xx[i] = np.dot(np.array([tx[i],ty[i]]).T,rotation_matrix(R))[0] + x
            yy[i] = np.dot(np.array([tx[i],ty[i]]).T,rotation_matrix(R))[1] + y
    
    return xx,yy,r1,r2

SHAPE = dict(
    rectangle=parameter_rectangle,
    circle=parameter_circle,
    triangle=parameter_triangle,
    ellipse=parameter_ellipse
    )

SHAPE_CHOICES = list(SHAPE.values())


def draw_shapes(st,n,shape=None):
    #ST为输入的总面积, N为生成的个数, shapes为固定形状的参数
    
    sm = st/n#平均面积
    
    k = 0#控制生成个数的循环的一个参数
    N = 720#画布大小的参数
    flag_1 = False
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})#生成figure与axis
    #用于判别重叠的矩阵的定义
    w, h = N, N
    Matrix = [[0 for x in range(w)] for y in range(h)]
    Matrix = np.array(Matrix)
        
    #判断重叠并实现绘图
    while k < n:
        x = np.random.randint(0,N) #图形的横坐标
        y = np.random.randint(0,N) #图形的纵坐标
        p = (1 - np.random.random())*(0.8) + 0.2
        angle = 2 * pi * np.random.random()
        
        #控制生成图形形状
        if shape is None:
            parameter_shape = np.random.choice(SHAPE_CHOICES)
        else:
            parameter_shape = SHAPE[shape]
        
        xx,yy,r1,r2 = parameter_shape(x,y,sm,p,angle)
        rr = int((r1**2+r2**2)**(0.5))
        if (max(xx+rr) >= N) | (max(yy+rr) >= N) | (min(xx-rr) <= 0) | (min(yy-rr) <= 0):
            continue
        
        overloping_judge_matrix = np.array([[i,j] for i in range(-r1,r1+1) for j in range(-r2,r2+1)]).T
        overloping_judge_matrix = np.dot(overloping_judge_matrix.T,rotation_matrix(angle))
        overloping_judge_matrix = overloping_judge_matrix.astype(int)
        overloping_judge_matrix = [[x,y] for ii in range(-r1,r1+1) for jj in range(-r2,r2+1)]+overloping_judge_matrix
        ojm = overloping_judge_matrix
        ojm = np.unique(ojm,axis=0)
        
        for zz in range(len(ojm)):
            if Matrix[ojm[zz][0],ojm[zz][1]] == 1:
                flag_1 = True
                for m in range(zz):
                    Matrix[ojm[m][0],ojm[m][1]] = Matrix[ojm[m][0],ojm[m][1]] - 1 
                break
            else:
                flag_1 = False
                Matrix[ojm[zz][0],ojm[zz][1]] = Matrix[ojm[zz][0],ojm[zz][1]] + 1
        if flag_1 == False:
            ax.fill(xx,yy)
            print(k)
            k = k + 1
        
#        for p in range(x-rr,x+rr)  :#获得围成该图形的最小矩形的“是否填充”的矩阵
#            for q in range(y-rr,y+rr):
#                if(Matrix[p,q] == 1):#如果有已被填充的，跳出p、q的循环，也不填充
#                    flag_1 = True
#                    for m in range(x-rr,p-1):
#                        for nn in range(y-rr,y+rr):
#                            Matrix[m,nn] = Matrix[m,nn] - 1
#                    for mm in range(y-rr,q):
#                        Matrix[p,mm] = Matrix[p,mm] - 1
#                    break
#                else:
#                    flag_1 = False #如果没被填充，则在填充矩阵里记录
#                    Matrix[p,q] = Matrix[p,q] + 1
#            if flag_1:
#                break
# 
#        if  flag_1 == False:
#            ax.fill(xx,yy) #如果整个图形都未被填充，则描绘它 
#            print(k)
#            k = k + 1
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    plt.show()
    
draw_shapes(90000,200)