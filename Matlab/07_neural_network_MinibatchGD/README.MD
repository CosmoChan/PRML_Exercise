 神经网络小批量梯度下降程序 wuweizhen version


 直接运行mnist_MinibatchGD_NN即可

 本程序使用按无放回顺序循环的小批量梯度下降，和之前的批量梯度下降相比，只做了很小的修改

 顺序选取小批量的细节详见NN_train，建议尝试使用不同的随机优化策略，例如一个随机抽取的方法是：

 在进行每个epoch时首先生成一个从1到样本大小的不重复的随机序列，以这个序列为索引，每次抽取batch size个mini-batch进行训练

 本程序默认仅使用迭代次数来控制停止，也建议尝试补充更合理的迭代停止条件

 由于小批量的方法具有一定的随机性，计算的损失函数值不稳定。可以尝试取最近n次的平均值来估计

 术语：
 Gradient Descent   : 梯度下降
 batch GD  		    ：批量梯度下降，batch size等于训练集大小
 mini-batch GD      : 小批量梯度下降，batch size介于1和训练集大小之间
 Stochastic GD(SGD) : 随机梯度下降，batch size等于1
 batch size         ：意为批大小，在每个iteration中从batch(训练集)中取出的mini-batch(训练子集)的大小。
 iteration 			：利用一个mini-batch进行一次训练，称为一个iteration(迭代)
 epoch  			：当batch中每个数据都进行了一次训练后，称为一个epoch