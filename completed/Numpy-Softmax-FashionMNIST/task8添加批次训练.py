import numpy as np
import torchvision
import os

"""
   数据初始化
"""
train_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=True,download=True)
X_train_raw=train_data.data.numpy()
X=X_train_raw.reshape(X_train_raw.shape[0],-1)/255.0

y=train_data.targets.numpy()

N=X.shape[0]
D=X.shape[1]
C=10

"""
   参数初始化
"""
W=np.random.randn(D,C)*0.01
b=np.zeros(C)

num_epochs=50
batch_size=64
learning_rate=0.1
num_batches=N//batch_size

"""
   函数定义
"""
def softmax(z):
    z_max=np.max(z,axis=1,keepdims=True)
    exp_z=np.exp(z-z_max)
    sum_exp_z=np.sum(exp_z,axis=1,keepdims=True)

    return exp_z/sum_exp_z

def cross_entropy_loss(y,probs):
    correct_probs=probs[np.arange(probs.shape[0]),y]

    return np.mean(-np.log(correct_probs))

"""
   训练循环
"""
for epoch in range(1,num_epochs+1):
    # 1.打乱数据
    indices=np.random.permutation(N)
    X_shuffled=X[indices]
    y_shuffled=y[indices]

    # 2.分成不同批次处理
    epoch_loss=0
    for batch in range(num_batches):
        start=batch*batch_size
        end=start+batch_size

        X_batch=X_shuffled[start:end]
        y_batch=y_shuffled[start:end]
        y_batch_onehot=np.eye(C)[y_batch]

        # 前向传播
        Z_batch = X_batch @ W + b
        probs_batch=softmax(Z_batch)
        loss_batch=cross_entropy_loss(y_batch,probs_batch)
        epoch_loss+=loss_batch

        # 反向传播计算梯度
        dZ=(probs_batch-y_batch_onehot)/batch_size
        dW=X_batch.T @ dZ
        db=np.sum(dZ,axis=0)

        # 参数更新
        W = W - learning_rate * dW
        b = b - learning_rate * db

    # 3.每个轮次结束后更细一次平均损失,每10轮打印一次损失和准确率
    avg_epoch_loss=epoch_loss/num_batches

    if epoch%10==0:
        Z_new = X @ W + b
        probs_new=softmax(Z_new)
        y_pred=np.argmax(probs_new,axis=1)
        accuracy=np.mean(y==y_pred)

        print(f"Epoch:{epoch} | Average Loss:{avg_epoch_loss:.4f} | Accuracy:{accuracy:.4f}")

print("Training Finished")

