import numpy as np

"""
   准备一个小型数据集
"""
# 5个样本，4个特征，3个类别
N,D,C=5,4,3

X=np.random.randn(N,D)
y=np.array([0,1,2,2,1])
y_onehot=np.eye(C)[y]

W=np.random.randn(D,C)
b=np.zeros(C)

"""
   前向计算
"""
Z = X @ W + b

def softmax(z):
    z_max=np.max(z,axis=1,keepdims=True)
    z_modified=z-z_max

    exp_z=np.exp(z_modified)
    sum_exp_z=np.sum(exp_z,axis=1,keepdims=True)

    probs=exp_z/sum_exp_z
    return probs

def cross_entropy_loss(y,probs):
    N=probs.shape[0]
    correct_probs=probs[np.arange(N),y]

    log_probs=-np.log(correct_probs)

    loss=np.mean(log_probs)
    return loss

probs=softmax(Z)
loss=cross_entropy_loss(y,probs)

print(f"loss={loss:.4f}")

"""
   解析梯度 
"""
dZ=(probs-y_onehot)/N
dW_analytical=X.T @ dZ
db_analytical=np.sum(dZ,axis=0)

print(f"dW_analytical={dW_analytical}")
print(f"db_analytical={db_analytical}")

"""
   数值梯度
"""
def compute_loss_given_W(W_given,X,b,y):
    Z=X @ W_given + b
    probs=softmax(Z)
    return cross_entropy_loss(y,probs)

epsilon=1e-5

dW_numerical=np.zeros_like(W)

print("Computing numerical gradient ...")
for i in range(D):
    for j in range(C):
        W_plus=W.copy()
        W_plus[i,j]+=epsilon
        loss_plus=compute_loss_given_W(W_plus,X,b,y)

        W_minus=W.copy()
        W_minus[i,j]-=epsilon
        loss_minus=compute_loss_given_W(W_minus,X,b,y)

        dW_numerical[i,j]=(loss_plus - loss_minus)/(2*epsilon)

"""
   比较两种梯度
"""
print(f"analytical gradient={dW_analytical}")
print(f"numerical gradient={dW_numerical}")
print(f"the largest difference:{np.max(np.abs(dW_analytical-dW_numerical)):.15f}")
print("Is the gradient analogous?",np.allclose(dW_analytical,dW_numerical,atol=1e-6))