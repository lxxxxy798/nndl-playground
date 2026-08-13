import torchvision
import os

"""
   加载 Fashion-MNIST数据集
"""
train_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=True,download=True)

test_data=torchvision.datasets.FashionMNIST(root=os.path.join('..','data'),train=False,download=True)

"""
   处理数据格式并验证
"""
X_train_raw=train_data.data.numpy()
X_test_raw=test_data.data.numpy()

X_train=X_train_raw.reshape(X_train_raw.shape[0],-1)
X_train=X_train/255.0

X_test=X_test_raw.reshape(X_test_raw.shape[0],-1)
X_test=X_test/255.0

y_train=train_data.targets.numpy()
y_test=test_data.targets.numpy()

print(f"X_train shape:{X_train.shape}")
print(f"X_train min:{X_train.min():.4f},max:{X_train.max():.4f}")
print(f"X_test shape:{X_test.shape}")
print(f"y_train shape:{y_train.shape}")
print(f"y_train top10:{y_train[:10]}")
print(f"y_test shape:{y_test.shape}")