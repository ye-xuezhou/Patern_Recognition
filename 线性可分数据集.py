#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#将数据保存成字典的形式
train_data_set = {-1:np.array([[1,1]]),
                 1:np.array([[3,3],
                             [4,3]])}
data_color={-1:'r',1:'b'}


# In[3]:


def train(data):
    rate=1  #学习速率
    w=np.array([0,0]) #斜率
    b=0  #截距
    optimize1=True
    while optimize1:
        a=0
        exit_flag1=True
        for i in data:           #i=1或-1
            for j in data[i]:
                if  i*(np.dot(j,w)+b) <=0:
                    w=w+rate*j*i
                    b=b+rate*i
                    
                    exit_flag1=False
                    break   #跳出内层for循环
                else:
                    a=a+1
            if not exit_flag1:
                break       #跳出外层for循环,进入内层的while循环
        
        if a==3:   #样本的容量,代表所需循环的次数,意思是只有当不再出现分类错误的现象才能退出while
            break
    return w,b


# In[4]:


w_t,b_t=train(train_data_set)
print("w0=",w_t[0])
print("w1=",w_t[1])
print("b=",b_t)

