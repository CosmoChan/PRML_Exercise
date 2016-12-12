function mnist_NN_MinibatchGD
% 神经网络小批量梯度下降程序 wuweizhen version
% 说明：
%    直接运行mnist_MinibatchGD_NN即可
%    本程序使用按无放回顺序循环的小批量梯度下降，和之前的批量梯度下降相比，只做了很小的修改
%    顺序选取小批量的细节详见NN_train，建议尝试使用不同的随机优化策略，例如一个随机抽取的方法是：
%    在进行每个epoch时首先生成一个从1到样本大小的不重复的随机序列，以这个序列为索引，每次抽取batch size个mini-batch进行训练
%    本程序默认仅使用迭代次数来控制停止，也建议尝试补充更合理的迭代停止条件
%    由于小批量的方法具有一定的随机性，计算的损失函数值不稳定。可以尝试取最近n次的平均值来估计
% 
% 术语：
%    Gradient Descent   ：梯度下降
%    batch GD  		    ：批量梯度下降，batch size等于训练集大小
%    mini-batch GD      ：小批量梯度下降，batch size介于1和训练集大小之间
%    Stochastic GD(SGD) ：随机梯度下降，batch size等于1
%    batch size         ：批大小，在每个iteration中从batch(训练集)中取出的mini-batch(训练子集)的大小。
%    iteration 			：利用一个mini-batch进行一次训练，称为一个iteration(迭代)
%    epoch  			：当batch中每个数据都进行了一次训练后，称为一个epoch
%
% 配置参数：
%    digits 是一个数字的向量，包含进行分类的目标数字
%    config 是一个向量，向量的维数控制神经网络的层数L，其各元素大小控制各层的但单元数量
%    activations 元胞数组，包含 L-1 个激活函数句柄，依此对应第2,3,...,L层的激活函数
%    derivatives 元胞数组，包含 L-1 个导函数的句柄，依此对应第2,3,...,L层的激活函数
%    cost_function 函数句柄，设置使用的损失函数
%    max_epochs 正整数，整个训练集的最大训练次数，本程序用最大迭代次数来控制迭代停止
%    eta 非负数，为学习速率。若设置太大，则损失函数将发生震荡，难以收敛，若太小，则学习效率低
% 输出：
%    默认在命令窗口输出训练时间和测试准确率。可以通过修改来输出
%
% 示例
%    对0-9进行分类，输入层有784个单元，一个有50个单元的隐藏层，输出层有10个单元。
%    隐藏层激活函数为tanh，输出层激活函数为softmax，学习速率为0.03，批大小为10，样本最大迭代次数为5次
%    程序将在1分钟内完成训练，并且测试正确率约为96%
%    参数的示例如下

% 参数配置
addpath functions

digits = [ 0 1 2 3 4 5 6 7 8 9 ];

struct = [ 784 50 10 ];

activations = { @tanh , @softmax };            %激活函数

derivatives = { @diff_tanh , @(x)1 };          %激活函数的导函数

cost_function = @cross_entropy;                %使用交叉熵损失函数

max_epochs = 5;                                %整个训练样本的迭代次数上限

batch_size = 10;                           %小批量梯度下降，每批的大小

learning_rate = 0.03;                          %学习速率

%加载训练数据和测试数据
[ train , test ] = load_mnist( digits );

%将输入矩阵X转置为每行一个样本，每列为一个维数
train.X = train.X';
test.X = test.X';

%将输出向量的标签y进行"one-of-K"编码，编码依据是digits向量
train.y = one_of_K( train.y , digits )';
test.y = one_of_K( test.y , digits )';

tic;
%进行训练，得到参数集合Wb，包含各层的W和b
Wb = NN_train( train.X , train.y , struct , activations , derivatives , cost_function , batch_size , max_epochs , learning_rate );
train_time = toc;

%在系数集合Wb下求输入矩阵test.X在网络的输出值
hat_Y = NN_test( test.X , Wb , activations );
    
%将网络的输出值矩阵正规化为one-of-K编码的标签矩阵
hat_T = regularize( hat_Y );
    
%计算分类正确率，存入列元胞accuracy的第 i 个位置
n = size( test.y , 1 );
accuracy = 1 - sum( sum( abs( test.y - hat_T ) ) ) / ( n * 2 );

fprintf('train time: %.2fs, accuracy: %.2f%%\n' , train_time , accuracy*100 );

%返回结果，去掉下一行的注释，并在最上方的function后面加上 result= ，即可输出各网络的各层参数
%result = [ Wb , accuracy ];

end