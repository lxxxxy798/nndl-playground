import numpy as np
import torchvision
import os
import matplotlib.pyplot as plt

"""
   数据初始化
"""
train_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=True,download=True)
test_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=False,download=True)

X_train_raw=train_data.data.numpy()
X=X_train_raw.reshape(X_train_raw.shape[0],-1)/255.0

X_test_raw=test_data.data.numpy()
X_test=X_test_raw.reshape(X_test_raw.shape[0],-1)/255.0

y=train_data.targets.numpy()
y_test=test_data.targets.numpy()

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
loss_history=[]
accuracy_history=[]

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

    # 3.每个轮次结束后更细一次平均损失,每5轮记录一次损失和准确率
    avg_epoch_loss=epoch_loss/num_batches

    if epoch%5==0:
        Z_new = X @ W + b
        probs_new=softmax(Z_new)
        y_pred=np.argmax(probs_new,axis=1)
        accuracy=np.mean(y==y_pred)

        loss_history.append(avg_epoch_loss)
        accuracy_history.append(accuracy)

        print(f"Epoch:{epoch} | Average Loss:{avg_epoch_loss:.4f} | Accuracy:{accuracy:.4f}")

print("Training Finished")

"""
   计算测试集上的准确率、精确率、召回率和F1值
"""
Z_test=X_test @ W + b
probs_test=softmax(Z_test)
y_pred_test=np.argmax(probs_test,axis=1)
accuracy_test=np.mean(y_pred_test==y_test)

print(f"Test Accuracy:{accuracy_test:.4f}")

"""
   绘制 损失曲线(红)与 准确率曲线(蓝)
"""
fig,ax1=plt.subplots(figsize=(8,5))

if len(loss_history)>0 and len(accuracy_history)>0:
    epochs_record=list(range(5,num_epochs+1,5))
    ax1.plot(epochs_record,loss_history,'r-',label="Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss",color="r")
    ax1.tick_params(axis="y",labelcolor="r",color="r")

    ax2=ax1.twinx()
    ax2.plot(epochs_record,accuracy_history,'b-',label="Training Accuracy")
    ax2.set_ylabel("Accuracy",color="b")
    ax2.tick_params(axis="y",labelcolor="b",color="b")

    fig.legend(loc="upper left")

    plt.title("Training Loss and Accuracy")
    plt.savefig("training_curves.png",dpi=150)
    plt.show()

"""
   绘制测试集上的预测标签与真实标签对比图(无放回随机抽取10个)
"""
indices=np.random.choice(X_test.shape[0],size=10,replace=False)
X_sample=X_test[indices]
y_sample=y_test[indices]
y_pred_sample=np.argmax(softmax(X_sample@W+b),axis=1)

fig,axes=plt.subplots(2,5,figsize=(15,8))
axes=axes.flatten()

for i in range(len(y_sample)):
    img=X_sample[i].reshape(28,28)
    axes[i].imshow(img,cmap="gray")
    axes[i].set_title(f"True Label:{y_sample[i]},Predicted Label:{y_pred_sample[i]}")
    axes[i].axis('off')

plt.figtext(0.56,0.03,"Label 0=T-shirt/top   Label 1=Trouser   Label 2=Pullover   Label 3=Dress   Label 4=Coat\n"
                     "Label 5=Sandal   Label 6=Shirt   Label 7=Sneaker   Label 8=Bag   Label 9=Ankle boot",
            ha="left",fontsize=10,fontstyle='italic',alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor='blue',linewidth=2))
plt.tight_layout()
plt.savefig("sample_predictions.png",dpi=150)
plt.show()