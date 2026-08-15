# NumPy-Softmax-FashionMNIST

## 项目背景

用纯 NumPy（不依赖 PyTorch/TensorFlow 等深度学习框架）从零实现 Softmax 回归模型，
在 Fashion-MNIST 数据集上训练并达到 85% 以上的测试准确率。

本项目是我在暑假期间自学深度学习基础时完成的，目的是通过手写反向传播和梯度下降，
建立对神经网络训练底层原理的具体理解。

## 数学原理

- **模型**：Softmax 回归（线性模型 + Softmax 激活 + 交叉熵损失）
- **前向传播**：`Z = X @ W + b` → Softmax → 概率分布
- **损失函数**：交叉熵损失 `L = -mean(log(probs[true_class]))`
- **反向传播**：手动推导并实现了交叉熵 + Softmax 的联合梯度 `dZ = (probs - y_onehot) / N`
- **优化算法**：Mini-batch 随机梯度下降，`batch_size=64`，`learning_rate=0.1`

## 项目结构

```text
NumPy-Softmax-FashionMNIST/
├── task1 加载数据.py
├── task2 实现 Softmax 函数.py
├── task3 实现交叉熵损失函数.py
├── task4 阶段一回顾.py
├── task5 实现前向传播.py
├── task6 实现反向传播的梯度计算.py
├── task7 实现不含批次的完整训练.py
├── task8 添加批次训练.py
├── task9 可视化.py
├── softmax_numpy.py
├── training_curve.png
└── sample_predictions.png
```

> 说明：`task*.py` 为分阶段拆解的练习脚本，按编号顺序逐个完成；`softmax_numpy.py` 为全部任务整合后的完整可运行版本。

## 运行方式

### 环境依赖

- Python 3.14
- NumPy
- Matplotlib
- torchvision（仅用于下载 Fashion-MNIST 数据集，不用于模型构建或训练）

### 运行步骤

1. 克隆本仓库
2. 安装依赖：`pip install numpy matplotlib torchvision`
3. 运行训练脚本：`python softmax_numpy.py`
4. 训练结束后，控制台会打印测试准确率，同时生成两张可视化图片

## 训练结果

- 最终训练准确率：**82.51–87.20%**
- 最终测试准确率：**82.83–84.53%**

> 注：由于权重随机初始化、Mini-batch 样本打乱，每次运行结果会存在浮动。以上范围为十次测试所得。

- 训练曲线：见 `training_curve.png`
- 预测结果样例：见 `sample_predictions.png`

## 个人收获

通过这个项目，我具体掌握了：

1. 从数学公式到 NumPy 代码的完整翻译过程
2. 反向传播中链式法则的实际计算（交叉熵 + Softmax 的联合梯度）
3. Mini-batch 梯度下降的训练循环实现
4. 模型评估（准确率、损失曲线）的基本方法

## 联系方式

西南大学 计算机与信息科学学院 软件学院

计算机科学与技术专业（中外合办） 陆熙悦

swulxxxxy@email.swu.edu.cn
