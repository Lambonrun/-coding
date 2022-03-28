# -*- coding: utf-8 -*-

from warnings import resetwarnings
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import random
import math

### 绘图函数 ###
def plot(accu,epo):
  plt.plot(epo,accu)
  plt.ylabel("accurancy")
  plt.xlabel("epoches")
  plt.show()
### 激活函数 ###
def ReLU(x):
  return np.maximum(x, 0)

def MSE(y_true, y_pred):
  return np.square(np.subtract(y_true, y_pred)).mean()

def CrossEntropy(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))

def LR_Adjuster(l,e):
  return l+0.5*math.sin(0.2*e*math.pi)

### 预测函数 ###
def PredictImage(img, w, b):
  resp = list(range(0, 10))
  for i in range(0,10):
    r = w[i] * img
    r = ReLU(np.sum(r) + b[i])
    resp[i] = r

  return np.argmax(resp)

### 检验函数 ###
def Predict(X, Y, w, b):
  total = len(test_images)
  valid = 0
  invalid = []

  for i in range(0, total):
    img = X[i]
    predicted = PredictImage(img, w, b)
    true = Y[i]
    if predicted == true:
      valid = valid + 1
    else:
      invalid.append({"image":img, "predicted":predicted, "true":true})
  return (float(valid) / total * 100)

### 训练函数 ###
def train(X,Y,learning_rate=0.5,epoch=10):

    ### 随机产生权重和偏置 ###
  w = (2 * np.random.rand(10, 784) - 1) / 10
  b = (2 * np.random.rand(10) - 1) / 10

    ### 用MSRA方法初始化权重和配置 ###
  #w = (2 * np.random.normal(0,math.sqrt(2/784),size=(10,784)) - 1) / 10
  #b = (2 * np.random.normal(0,math.sqrt(2/784),size=10) - 1) / 10
  
  accu=[]
  epo=[]

  for e in range(epoch+1):
    for n in range(len(X)):
    #for n in random.sample(range(len(X)),k=54000):#伪dropout正则化
      img = X[n]
      cls = Y[n]

        ### 显示每个图的概率向量 ###
      resp = np.zeros(10, dtype=np.float32)
      for i in range(0,10):
        r = w[i] * img
        r = ReLU(np.sum(r) + b[i])
        resp[i] = r

        '''
        找到最大概率值；我们把剩下的数组值和最大替换概率为1
        '''
      resp_cls = np.argmax(resp)
      resp = np.zeros(10, dtype=np.float32)
      resp[resp_cls] = 1.0

        ### 求出数字 ###
      true_resp = np.zeros(10, dtype=np.float32)
      true_resp[cls] = 1.0
        #########################################
      error=MSE(true_resp,resp)
      '''
        找出误差并计算新的权重系数值和新的位移系数值
        '''
      
      error = resp - true_resp
      delta = error * ((resp >= 0) * np.ones(10))

      for i in range(0,10):
        w[i] -= learning_rate*np.dot(img, delta[i])
        b[i] -= learning_rate*delta[i]
    
    Ac=Predict(test_images,test_labels,w,b)
    accu.append(Ac)
    epo.append(e)
    #learning_rate=LR_Adjuster(0.5,e) #学习率调整
  #print(max(accu))
  return accu,epo
  #  print(epoch,"done")



if __name__ ==  '__main__':
  '''
  获取数据转化为一维数组
  '''
  mndata = MNIST("./data")
  tr_images, tr_labels = mndata.load_training()
  test_images, test_labels = mndata.load_testing()

  ### 图片转换 ###
  for i in range(0, len(test_images)):
    test_images[i] = np.array(test_images[i], dtype="float") / 255
      
  for i in range(0, len(tr_images)):
    tr_images[i] = np.array(tr_images[i], dtype="float") / 255

  accu,epo=train(tr_images,tr_labels,0.5,1)
  plot(accu,epo)

