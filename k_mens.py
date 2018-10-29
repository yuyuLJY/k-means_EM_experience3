# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt


def loan_data(txt_name):
    '''
    读入数据
    :param txt_name:
    :return:数据矩阵
    '''
    data = open(txt_name).readlines()
    length = len(data)
    list_tolist = []
    for line in data:
        line = line.strip().split(' ')
        for i in range(2):
            line[i] = float(line[i])
        list_tolist.append(line)
    return np.mat(list_tolist)

def k_means(dataSet, k):
    '''
    k-means算法的实现
    dataSet-数据集, k-聚类中心的个数, disMeas-衡量距离的算法类型，createCent-初始点的选取
    :param dataSet:
    :param k:
    :return:聚类的结果ClustDist(第一列：属于哪一个类index 第二列：到index的距离)，聚类的中心点
    '''
    row ,col = dataSet.shape
    ClustDist = np.mat(np.zeros((row, 2)))
    clustercents = randCent(dataSet,k) # 得到初始化的聚类中心(是一个k*col的矩阵)
    flag = True
    while flag:
        for i in range(row): # 对每一个点遍历
            # 初始化某个点的最小距离、某个点属于哪一个的聚类中心
            minDist = float("inf")
            minIndex = -1
            for j in range(k): # 对每一个聚类圈遍历
                dist = distEclud(clustercents[j,:],dataSet[i,:])
                if dist < minDist:
                    minDist = dist
                    minIndex = j  # 记下这个类别的标签
            # 每次遍历完一个结点，都要判断是否收敛
            if ClustDist[i, 0] == minIndex:  # 不再找到一个新聚类中心
                flag = False  # 不再迭代
            ClustDist[i, :] = minIndex, minDist
        print("该轮结束情况",ClustDist)
        # 所有点遍历完毕，重新计算k个聚类中心
        for cent in range(k):
            ClustDist_toarry = ClustDist.A # 把聚类中心的矩阵变成array
            # 第一列跟cent比较，选取出属于cent类的行坐标，并且记下坐标
            cent_index = np.nonzero(ClustDist_toarry[:, 0] == cent)[0]  # 得到同一个聚类的所有索引
            cent_data = dataSet[cent_index]
            clustercents[cent,:] = np.mean(cent_data, axis=0)  # 对列求均值
    return clustercents,ClustDist

def randCent(dataSet, k):
    '''
    #随机选取初始向量
    :param dataSet:
    :param k:
    :return: 选取的初始向量
    '''

    '''
        clustercents = np.mat([[-0.035023111364824104 ,0.024620409368181342],
         [-0.19277372471361123, 0.5012183691784274],
         [1.3959593148960707 ,-0.03762736346583746],
         [1.3024832171337226 ,-1.3778977721666346]])
         return clustercents
    '''
    row,  col = dataSet.shape
    clustercents = np.mat(np.ones((k,col))) #创建 k*col(4*2)行的初始矩阵
    # 把初始矩阵用随机数填满
    for i in range(col):
        min_colnumber = min(dataSet[:,i])
        max_colnumber = max(dataSet[:,i])
        clustercents[:,i] = np.mat(np.random.rand(k,1)*float(max_colnumber-min_colnumber)+min_colnumber)
    return clustercents

def distEclud(vecA, vecB):
    '''
    #衡量距离的算法
    :param vecA:
    :param vecB:
    :return: 两个向量的距离
    '''
    return np.linalg.norm(vecA-vecB)

def draw_picture(clustercents, ClustDist,k,dataSet):
    '''
    根据传入的k个聚类的中心点、每个聚类的数据情况，画图
    :param clustercents:
    :param ClustDist:
    :param k:
    :return:
    '''
    ClustDist_toarray = ClustDist.A
    dataSet_toarray = dataSet.A
    plt.title('k=4',fontsize=20)
    # index =[] #记录下每一个聚类中心的数据的索引
    color = ['blue','yellow','green','black','c'] # g’‘b’‘c’‘m’‘y’‘k’
    for cent in range(k):
        index = np.nonzero(ClustDist_toarray[:,0]==cent)[0]
        cluster = dataSet_toarray[index]
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color[cent], marker='o')
        plt.scatter(clustercents[cent, 0], clustercents[cent, 1],c='red', marker='x')
    plt.show()

def main():
    '''
    整体流程：（1）先读入数据。
             （2）再使用k-means算法：随机选取一个中心矩阵，计算每一个点到中心矩阵的距离，更新中心矩阵。
             （3）画图
    '''
    k = 4
    dataSet = loan_data("data.txt")
    # print(dataSet) #loanData成功
    # random = randCent(dataSet,k)
    # print(random) # 随机产生的矩阵成功
    clustercents, ClustDist = k_means(dataSet,k)
    print('聚类中心',clustercents)
    print('分类情况',ClustDist)
    draw_picture(clustercents, ClustDist,k,dataSet)

if('main'=='main'):
    main()