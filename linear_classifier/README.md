# MNIST 手写数字识别数据集介绍
手写数字识别数据集官方网站是http://yann.lecun.com/exdb/mnist/, 上面有关于这份数据集很详尽的介绍.

这里简要介绍一下: 这份数据集的训练集含有6万个样本, 测试集含有1万个样本, 图片的大小经过了标准化($28\times28$像素), 图中的数字进行了居中处理. 注意这是一份真实数据集来的.

官网下载下来的数据集包含4个文件, 分别是
* 训练集图像train-images-idx3-ubyte.gz
* 训练集标签train-labels-idx1-ubyte.gz
* 测试集图像t10k-images-idx3-ubyte.gz
* 测试集标签t10k-labels-idx1-ubyte.gz

注意测试集中前5000个样本是来自原NIST的训练集, 而后5000个样本是来自原NIST的测试集, 所以前5000个样本比后5000个样本数据更干净, 更简单.

datasets文件夹的loadMNISTImages.m和loadMNISTLabels.m文件中分别给出了读取这份数据集的图像和标签的m文件, 使用时可直接调用这两个文件而不必担心读取的实现细节.

# 训练集测试集载入
本文件夹中的load__mnist文件可加载MNIST训练集和测试集, 把加载的图像放入矩阵X中, 使得第i张图像的第j个像素是X中的元素$X__{ji}$. 同时把标签载入变量y中. 此外, 还对像素灰度做了简单的标准化工作. load___mnist函数接受一个参数binary__digits, 这是一个bool值, 若为真则加载的数据只使用了0, 1两个数字, 若为false则加载0-9所有数字. 默认为true.
