import numpy as np
"""
   输入概率矩阵和真实标签
"""
N=int(input())

y=np.random.randint(0,10,(N,))
print(f"真实标签:{y}")

z=np.random.randn(N,10)
z_max=np.max(z,axis=1,keepdims=True)
z_modified=z-z_max

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)

probs=softmax(z_modified)
print(f"概率矩阵:",probs)

"""
   实现交叉熵损失函数
"""
def cross_entropy_loss(y,probs):
    N=probs.shape[0]

    correct_probs=probs[np.arange(N),y]
    log_probs=-np.log(correct_probs)

    loss=np.mean(log_probs)
    return loss

"""
   计算验证
"""
loss=cross_entropy_loss(y,probs)
print(f"交叉熵损失:{loss:.4f}")

manual_loss=-np.log(probs[0,y[0]])
print(f"手动计算第一个样本的损失:{manual_loss:.4f}")