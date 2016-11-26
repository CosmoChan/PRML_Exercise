function result = mnist_NN
% 神经网络实验调度程序
%
% 参数：
%    digits 是一个数字的向量，包含进行分类的目标数字
%    config 是一个向量，向量的维数控制神经网络的层数L，其各元素大小控制各层的但单元数量
%    activations 元胞数组，包含 L-1 个激活函数句柄，依此对应第2,3,...,L层的激活函数
%    derivatives 元胞数组，包含 L-1 个导函数的句柄，依此对应第2,3,...,L层的激活函数
%    max_iterations 正整数，是最大迭代次数，本程序用最大迭代次数来控制迭代停止
%    eta 非负数，为学习速率。若设置太大，则损失函数将发生震荡，难以收敛，若太小，则学习效率低
% 输出：
%    result 是一个元胞数组，包含
%         Wb 训练所得到的参数集合 W , b
%         accuracy 测试正确率
% 示例
%    对0-9进行分类，输入层有784个单元，一个有100个单元的隐藏层，输出层有10个单元。
%    隐藏层激活函数为tanh，输出层激活函数为sigmoid，学习速率为1，进行200次的迭代，
%    程序参数的示例如下

% 参数配置
addpath Activations_functions

digits = [ 0 1 2 3 4 5 6 7 8 9 ];

config = [ 784 100 10 ];

activations = { @tanh , @sigmoid };

derivatives = { @diff_tanh , @diff_sigmoid };

max_iterations = 200;

eta = 2;

%加载训练数据和测试数据
[ train , test ] = load_mnist( digits );

%将输入矩阵X转置为每行一个样本，每列为一个维数
train.X = train.X';
test.X = test.X';

%将输出向量的标签y进行"one-of-K"编码，编码依据是digits向量
train.y = one_of_K( train.y , digits )';
test.y = one_of_K( test.y , digits )';

%进行训练，得到参数集合Wb，包含各层的W和b
Wb = NN_train( train.X , train.y , config , activations , derivatives , max_iterations , eta );
   
%在系数集合Wb下求输入矩阵test.X在网络的输出值
hat_Y = NN_test( test.X , Wb , activations );
    
%将网络的输出值矩阵正规化为one-of-K编码的标签矩阵
hat_T = regularize( hat_Y );
    
%计算分类正确率，存入列元胞accuracy的第 i 个位置
n = size( test.y , 1 );
accuracy = 1 - sum( sum( abs( test.y - hat_T ) ) ) / ( n * 2 );

result = [ Wb , accuracy ];

end