import numpy as np
import os
import torchvision

"""
   初始化权重和偏置
"""
W=np.random.randn(784,10)*0.01
b=np.zeros(10)

"""
   导入输入矩阵
"""
train_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=True,download=True)
X_raw=train_data.data.numpy()
X_all=X_raw.reshape(X_raw.shape[0],-1)/255.0

X=X_all[:100,]
print(f"the shape of X is {X.shape}")

"""
   前向计算得到得分矩阵
"""
Z = X @ W + b
print(f"the shape of Z is {Z.shape}")

"""
   通过Softmax映射为概率分布
"""
def softmax(Z):
    Z_max=np.max(Z,axis=1,keepdims=True)
    Z_modified=Z-Z_max

    exp_Z=np.exp(Z_modified)
    divisor=np.sum(np.exp(Z_modified),axis=1,keepdims=True)

    probs=exp_Z/divisor

    return probs

probs=softmax(Z)
print(f"the shape of probs is {probs.shape}")