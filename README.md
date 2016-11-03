# PRML算法实现
## 简介

[Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/) by C.Bishop(PRML)中算法的Matlab及Python实现

## 描述

文件结构为

```
|-Matlab
	|_01_least_squares_classifier
	|_02_fisher_discriminant_analysis
	|_03_perceptron
	|_04_logistic_regression
	|_05_softmax_regression_classifier
	... ... ... 
|-Python
	|_01_least_squares_classifier
	|_02_fisher_discriminant_analysis
	|_03_perceptron
	|_04_logistic_regression
	|_05_softmax_regression_classifier
	... ... ...
|-datasets
|-README.md
```

有些章节中存在不同的version为不同成员写的代码, 可挑选自己喜欢的版本查看.

算法程序尽可能做到:

- 模块化
  大致按照以下分类

  - 算法**训练**部分
  - 算法**测试**部分
  - 算法在数据集(比如MNIST手写数字)上的应用

  为方便日后扩展时会另外编写辅助函数, 如对基函数文件(方便更换基函数)

- 高可读性

  - 详细的函数说明(Python版本的函数说明与sklearn等机器学习库保持一致)
  - 主函数中按步骤分隔, 并作必要的解释, 步骤如:
    - 加载训练数据
    - 预处理
    - 模型训练
    - 加载测试数据
    - 使用模型作预测
    - 预测表现计算
  - 少用降低可读性的缩写命名, 符号尽量与书中保持一致
  - 添加`txt`格式或`md`格式的文件说明

- 高效
  在保证可读性的同时, 尽可能使用高效的实现方式.如使用矩阵向量运算代替`for`循环

- 易于使用
  只需把数据放入`datasets`文件夹中即可运行所有程序.

# 如何使用

## 方式一

1. 点击`Clone or download`
2. 选择`Download ZIP`
3. [在此](http://yann.lecun.com/exdb/mnist/)下载MNIST手写数字数据集, 放入`datasets`文件夹中
4. 若日后有更新需要重新下载

## 方式二(推荐)

1. 下载[Git Bash](https://git-for-windows.github.io/)

2. 点击`Clone or download`

3. 复制该网址https://github.com/CosmoChan/PRML_Exercise.git

4. 在你希望存放本文件夹的地方打开命令行, 输入

   ```shell
   git clone https://github.com/CosmoChan/PRML_Exercise.git
   ```

5. 下载完成会出现本文件夹, [在此](http://yann.lecun.com/exdb/mnist/)下载MNIST手写数字数据集, 放入`datasets`文件夹中

6. 日后有更新直接在该文件夹中输入

   ```shell
   git pull
   ```

   即可

# 联系我们

有任何问题欢迎在`issue`中提出