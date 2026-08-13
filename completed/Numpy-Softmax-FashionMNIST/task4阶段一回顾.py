import numpy as np
import torchvision
import os

train_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=True,download=True)
test_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=False,download=True)

X_train_raw=train_data.data.numpy()
X_test_raw=test_data.data.numpy()

X_train=X_train_raw.reshape(X_train_raw.shape[0],-1)/255.0
X_test=X_test_raw.reshape(X_test_raw.shape[0],-1)/255.0

y_train=train_data.targets.numpy()
y_test=test_data.targets.numpy()

print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)
print("y_train top 10:",y_train[:10])

def softmax(z):
    z_max=np.max(z,axis=1,keepdims=True)
    z_modified=z-z_max

    exp_z=np.exp(z_modified)
    divisor=np.sum(np.exp(z_modified),axis=1,keepdims=True)

    probs=exp_z/divisor
    return probs

def cross_entropy_loss(y,probs):
    N=probs.shape[0]

    correct_probs=probs[np.arange(N),y]

    log_probs=-np.log(correct_probs)

    loss=np.mean(log_probs)

    return loss