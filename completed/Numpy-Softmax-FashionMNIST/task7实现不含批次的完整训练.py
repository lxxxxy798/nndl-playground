import numpy as np
import torchvision
import os

"""
   初始化
"""
# 输入
train_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=True,download=True)
X_train_raw=train_data.data.numpy()
X=X_train_raw.reshape(X_train_raw.shape[0],-1)/255.0

# 标签
C=10
y=train_data.targets.numpy()
y_onehot=np.eye(C)[y]

# 权重和偏置
W=np.random.randn(X.shape[1],C)*0.01
b=np.zeros(C)

"""
   前向传播 计算损失
"""
# 得分
Z= X @ W + b

# 通过Softmax函数映射为概率
def softmax(z):
    z_max=np.max(z,axis=1,keepdims=True)
    exp_z=np.exp(z-z_max)
    sum_exp_z=np.sum(exp_z,axis=1,keepdims=True)

    probs=exp_z/sum_exp_z
    return probs

probs=softmax(Z)

# 通过交叉熵损失函数计算损失
def cross_entropy_loss(y,probs):
    correct_probs=probs[np.arange(probs.shape[0]),y]
    loss=np.mean(-np.log(correct_probs))
    return loss

loss=cross_entropy_loss(y,probs)

"""
   反向传播计算梯度
"""
dZ=(probs-y_onehot)/X.shape[0]
dW=X.T @ dZ
db=np.sum(dZ,axis=0)

"""
   更新参数
"""
learning_rate=0.1
W=W-learning_rate*dW
b=b-learning_rate*db

"""
   验证
"""
Z_new=X @ W + b
probs_new=softmax(Z_new)
loss_new=cross_entropy_loss(y,probs_new)

y_pred=np.argmax(probs_new,axis=1)
accuracy=np.mean(y_pred == y)

print(f"loss after training: {loss_new:.4f}")
print(f"accuracy after training: {accuracy:.4f}")
print(f"loss descends:{loss-loss_new:.4f}")