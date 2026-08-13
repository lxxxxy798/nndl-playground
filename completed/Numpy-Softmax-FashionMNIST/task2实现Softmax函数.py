import numpy as np

"""
   创建得分矩阵
"""
N=int(input())
z=np.random.randn(N,10)

# 减去每行最大值保证数值稳定性
z_max=np.max(z,axis=1,keepdims=True)
z_modified=z-z_max
print(z_modified)
print("\n\n")

"""
   通过 Softmax映射为概率分布
"""
def softmax(z):
    probs=np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)
    return probs

"""
   验证
"""
result=softmax(z_modified)
print(result)

row_sum=np.sum(result,axis=1)
print(row_sum-1.0<0.01)
print(np.allclose(row_sum,1.0))