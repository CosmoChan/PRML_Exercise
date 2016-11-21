function result = mnist_NN( digits , config , activations , derivatives , max_iterations , eta )
% 神经网络实验调度程序
%
% 输入：
%    digits 是一个数字的向量，包含进行分类的目标数字
%    config 是一个向量，向量的维数控制神经网络的层数L，其各元素大小控制各层的但单元数量
%    activations 元胞数组，包含 L-1 个激活函数句柄，依此对应第2,3,...,L层的激活函数
%    dervatives 元胞数组，包含 L-2 个导函数的句柄，依此对应第2,3,...,L-1层的激活函数
%    max_iterations 正整数，是最大迭代次数，本程序用最大迭代次数来控制迭代停止
%    eta 非负数，为学习速率。若设置太大，则损失函数将发生震荡，难以收敛，若太小，则学习效率低
% 输出：
%    result 是一个元胞矩阵，
%        第一列是
%            Wb_LIST 有max_iterations行，第i行是第 i 次迭代产生的各层的系数 W 和 b
%        第二列是
%            accuracy 有max_iterations行，第i行是第 i 次迭代产生的系数集合的测试正确率
%
% 示例
%    对0-9进行分类，输入层有784个单元，一个有100个单元的隐藏层，输出层有10个单元。
%    隐藏层和输出层的激活函数为sigmoid函数，隐藏层导函数为diff_sigmoid函数。进行50次的迭代，
%    学习速率为0.7，程序调用的示例如下
%
%    >> Activations = { @sigmoid , @sigmoid };
%    >> Dervatives = { @diff_sigmoid };
%    >> Digits = [ 0 1 2 3 4 5 6 7 8 9 ];
%    >> Config = [ 784 100 10 ];
%    >> Max_iterations = 1000;
%    >> Eta = 0.7;
%    >> Result = mnist_NN( Digits, Config, Activations, Dervatives, Max_iterations, Eta )

%加载训练数据和测试数据
[ train , test ] = load_mnist( digits );

%将输入矩阵X转置为每行一个样本，每列为一个维数
train.X = train.X';
test.X = test.X';

%将输出向量的标签y进行"one-of-K"编码，编码依据是digits向量
train.y = one_of_K( train.y , digits )';
test.y = one_of_K( test.y , digits )';

%进行训练，该步骤将参数若干个结果，将结果存入系数列表Wb_LIST
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