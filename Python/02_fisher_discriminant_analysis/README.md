# Fisher Discriminant Analysis on MNIST dataset

# 简介

费舍尔线性判别法的模块化实现, 并在MNIST数据集上对数字6和8进行分类. 如果一切进行顺利, 你将看到`The error rate is : 0.0248`的提示

# 文件结构

`python-mnist` : 读取mnist数据集的python库源码

`FDA.py` : 包含FDA训练函数`FDA_train` 以及FDA测试函数`FDA_test`

`load_dataset`: 包含读取文件的`load_mnist`函数以及进行标准化的`normalization`函数

`mnist_FDA` : FDA在mnist数据上实现的**主程序**.

# 如何运行

代码写于python2.7,但应该可以兼容python3. 主要用到`numpy`库.鉴于以后可能使用到其他有关科学计算的python库, 建议安装[Anaconda](https://www.continuum.io/downloads),这会包含大部分常用的科学计算库, 让你免于安装一个个python库.

由于读取mnist数据时使用了python的mnsit库, 你可通过一下命令方便地安装

```bash
pip install python-mnist
```

或者在PRML_Exercise/Python/02_fisher_discriminant_analysis/version\_01/python-mnist文件夹中运行

```shell
python setup.py install
```

进行安装.

运行程序则输入

```shell
python mnist_FDA.py
```

若你使用的是Ipython也可输入

```shell
ipython mnsit_FDA.py
```

你可更改mnist_FDA.py中的变量y\_1, y\_2决定对哪两个数字进行分类.