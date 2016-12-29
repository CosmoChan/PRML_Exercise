# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pandas as pd
import random
import math
from numpy.matlib import repmat
from scipy.stats import zscore
import os
import time

start = time.clock()

def k_means(data, str_,  K):
    """
    k-means clustering
     
    Parameters
    ----------
    data : ndarray of shape (width,height,3) or (d,n) 
    str_ : string,表示要聚类的类型，image表示图片，old_faithful 表示老忠实数据
    
    
    Returns
    -------
   
    
    """
    print('=====================================================================')
    if str_ == 'image':
        print('K=%s时的图片分割的结果'%K)
        
    else :
        print('K=%s时的老忠实数据聚类的结果'%K)
    
    # 创建文件夹保存数据和图片
    CurrDir = os.getcwd()
    child_path = time.strftime("%Y%m%d"+"T"+"%H%M%S")
    path = CurrDir + "\\" + child_path
    # 创建文件夹
    os.mkdir(path)
    
    # 定义最大迭代次数iteration
    iteration = 200
    # 定义迭代停止条件
    tolerance = 1e-2
    
    
    # 考虑是图片的情况
    if str_ == 'image':
        # 画出原始的图像
        h = plt.figure()
        plt.title('K_means Raw picture(K=%s)'%K)
        io.imshow(np.uint8(data))       
        width = data.shape[0]
        height = data.shape[1]

        data = np.append(np.append(data[:,:,0].reshape(1,width*height),
                                  data[:,:,1].reshape(1,width*height),axis = 0),
                        data[:,:,2].reshape(1,width*height),axis = 0)

    
    # 考虑是老忠实喷泉的情况
    else :
        # 画出原始的图
        h = plt.figure()
        plt.plot(data[0,:],data[1,:],'k.')
        plt.title('K_means Raw data(K=%s)'%K)
        plt.show()  
    
    # 将图片保存到文件夹，并以“rawdata.jpg”命名
    picture_full_path = path + "\\rawdata.jpg"
    h.savefig(picture_full_path,format = 'jpg') 
    
    # 计算data的形状
    d,n = data.shape
    
    # 随机选择K个点作为聚类中心，但是这K个点不能使相同的
    list_ = range(0,n)
    idx_mu = random.sample(list_,K)
    # 选取聚类中心
    mu = data[:,idx_mu]
    
    # 聚类(M-step)
    dis = np.zeros([K,n])
    for i in range(K):
        dis[i,:] = np.sum((data - repmat(mu[:,i].reshape(d,1),1,n)) ** 2,0)
    idx = np.argmin(dis,0)
    r = np.zeros([K,n])
    for i in range(n):
        r[idx[i],i] = 1
    
    
    # 考虑是图片的情况
    if str_ == 'image':
        new_data = np.zeros([d,n])
        for i in range(K):
            new_data[:,r[i,:]==1] = mu[:,i].reshape(d,1)
        
        new_img = np.zeros([width,height,3])
        for i in range(3):
            new_img[:,:,i] = (new_data[i,:]).reshape(width,height)
        h = plt.figure()
        plt.title('K_means Random initial mean(K=%s)'%K)
        io.imshow(np.uint8( new_img))
        
    else:
        h = plt.figure()
        for i in range(K):
            plt.plot(mu[0,i],mu[0,i],'o',
                     data[:,r[i,:]==1][0,:],data[:,r[i,:]==1][1,:],'.')
            plt.title('K_means Random initial mean(K=%s)'%K)
            plt.show
    
    # 保存图片，将图片保存在路径path下，并以“random_initial.jpg”命名
    picture_full_path = path + "\\random_initial.jpg"
    h.savefig(picture_full_path,format = 'jpg')
      
    # 计算代价函数的初始值
    J = []
    J_ = 0
    for i in range(K):
        J_ += np.sum(repmat(r[i,:],d,1) * ((data - repmat(mu[:,i].reshape(d,1),1,n))**2))
    J.append(J_)
    
    # 进行迭代，直到到达最大迭代次数或达到迭代终止条件为止
    for j in range(iteration):
        # 计算新的聚类中心
        #################################################################################


        
        #################################################################################

        # 计算每个数据点到K个聚类中心的距离
        dis = np.zeros([K,n])
        for i in range(K):
            dis[i,:] = np.sum((data - repmat(mu[:,i].reshape(d,1),1,n)) ** 2,0)
 
        # 对dis的每一列检索最小值，并且将最小值的位置保存在idx中
        idx = np.argmin(dis,0)
        
        # 计算标签矩阵，将dis的每一列最小值的位置标记为1，其他标记为0
        r = np.zeros([K,n])
        for i in range(n):
            r[idx[i],i] = 1   

        if str_ == 'image':
            new_data = np.zeros([d,n])
            for i in range(K):
                new_data[:,r[i,:]==1] = mu[:,i].reshape(d,1)

            new_img = np.zeros([width,height,3])
            for i in range(3):
                new_img[:,:,i] = (new_data[i,:]).reshape(width,height)
            h = plt.figure()
            plt.title('K_means Iteration No.%s(K=%s)'%((j+1),K))
            io.imshow(np.uint8( new_img))

        else:
            h = plt.figure()
            for i in range(K):
                plt.plot(mu[0,i],mu[1,i],'o',
                         data[:,r[i,:]==1][0,:],data[:,r[i,:]==1][1,:],'.')
                plt.title('K_means Iteration No.%s(K=%s)'%((j+1),K))
                plt.show
                
        # 保存图片，将图片保存在路径path下，并以“j.jpg”命名
        picture_full_path = path + "\\" + str(j+1) + ".jpg"
        h.savefig(picture_full_path,format = 'jpg')
        
        # 计算损失函数J
        J_ = 0
        for i in range(K):
            J_ += np.sum(repmat(r[i,:],d,1) * ((data - repmat(mu[:,i].reshape(d,1),1,n))**2))
        J.append(J_)
        
        # 判断迭代是否终止，如果损失函数变化小于tolerance，则停止迭代，否则继续迭代
        if np.abs(J[j] - J_) < tolerance:
            break
        
    print('=====================================================================')
    if j == (iteration - 1):
        print('已经达到最大迭代次数，cost仍未收敛')
    else:
        print('迭代了%s次，cost达到收敛条件'%(j+1))
    
    
    # 画出代价函数J的变化曲线图
    h = plt.figure()
    plt.plot(range(len(J)),J,'-')
    plt.xlabel('Iteration NO.')
    plt.ylabel('Cost')
    plt.title('The Change of Cost with the Number of Iteration(K=%s)'%K)
    plt.show()
    # 将图片保存在路径path下，并以“change_of_J.jpg”命名
    picture_full_path = path + "\\change_of_J.jpg"
    h.savefig(picture_full_path,format = 'jpg')
    
    # 保存mu
    mu_full_path = path + "\\mu.txt"
    np.savetxt(mu_full_path,mu)
    # 读取mu
    #mu = np.loadtxt(mu_full_path)
    
    print('====================================================================')
    print('原始数据、图片、模型参数都保存在路径%s中'%path)
    print('\n\n\n\n\n\n\n')

if __name__ == "__main__":
    """
    
    主程序
    ======
    这个程序使用k-means算法，可以对离散数据old_faithful进行聚类，或者对图片进行聚类分割
    
    每次运行程序，程序会在当前路径生成一个当前时间的文件件，如20161220T225845，程序运行
    产生的数据、图片都会保存在文件夹中，方便查看、重现
    
    k_means
    =======
    k_means(data,str_,K)
    
    Parameters
    ----------
    @param data : ndarray of shape (width,height,3) or (d,n) 
    @param str_ : string,表示要聚类的类型，image表示图片，old_faithful 表示老忠实数据
    
    
    Returns
    -------   
    
    
    math
    ====
    @math： 目标函数
            $$J = \sum\limits_{n=1}^N\sum\limits_{k=1}^K \lVert x_n -\mu_k \rVert ^2$$
            
    @math： E-step
            $$ \mu_k = \frac{\sum_n r_{nk} x_n}{\sum_n r_{nk}} $$
            
    @math： M-step
            $$ r_{nk}=\begin {cases} 
                        1, & \text{如果k = $arg min_j||x_n-\mu_j||^2$} \\\ 
                        0, & \text{其他} 
                      \end {cases} $$
                      
                      
    运行
    ====
    在所需要的包都满足的情况下，在CMD命令窗口输入python k_means.py 
    或者在ipython 中输入%run k_means.py
    
    注意
    ====
    运行程序时需要对程序中导入的图片的名字进行修改，对于离散数据的名字也需要进行修改
    
    
    """

    
    ###########################################################################
    '''
    老忠实数据实验
    
    数据说明
    =======
    老忠实实验数据保存在old_faithful.csv文件中，其中包含两列数据，分别是
    eruptions（喷发时间）和waiting（等待时间），一共有272个样本
    
    
    实验结果
    =======
    当K=2时，数据聚类的结果稳定，每次运行都会形成两个稳定的聚类簇，因为从数据的
    图像上看，数据更倾向于聚成左下角和右上角两类
    当K=3时，数据聚类的结果不稳定，每次运行的结果可能不一样，有时候数据的左下角
    有两个聚类簇，有时候左下角有一个聚类簇，这与均值点的随机初始化有关
    当K=4时，结果与K=3是很像，左下角和右上角的聚类中心的比值可能是1:3,2:2,3:1
    '''
    
    # 导入数据
    data = pd.read_csv('old_faithful.csv')
    # 提取数据
    data = data[data.columns[0]].str.split().apply(pd.Series,1).astype('float64')
    del data[0]
    data.columns = (0,1)
    data = np.array(data).T
    
    # 数据归一化处理
    data = zscore(data,axis = 1)
    
    # 调用k_means函数
    for K in range(2,5):
       k_means(data,'old_faithful',K)
            
    ###########################################################################    
    '''
    图片聚类分割实验
    
    数据说明
    =======
    one.jpg 是一张570*760的图片，导入图片，将图片保存为570*760*3的RGB数据
    
    实验结果
    =======
    K越大，实验结果的图片与原图相近度更高
    
    
    ''' 
    # 导入图片数据，记得改名字！！！！！！！！！！！！！！！！！！！！！！
    data = io.imread('one.jpg')
    
    # 调用k_means函数
    for K in range(2,5):
        k_means(data,'image',K)
        
    end = time.clock()
    print('====================================================================')
    print('程序共用时间：%s'%(end-start))