function result = softmax_mnist( digits , lambda )
% softmax_mnist:
%     基于softmax函数的多类logistic回归的实现程序。利用Newton-Raphson迭代最优
%     化框架进行求解。
% 输入：
%     digits 是一个K维向量，包含K个不相同的目标数字
%     lambda 是一个非负数，为正则化系数
% 输出：
%     result 是一个元胞，包含：
%          W 为d行K列的矩阵，模型参数矩阵，每一列都是digits中对应分类的参数
%          norm(W) 是参数矩阵W的二范数
%          iterations 是训练器的迭代次数
%          accuracy 是训练模型在给定测试数据下的正确率
%
% 实例：
%      输入：
%          >> softmax_mnist( [ 0 1 2 3 4 5 6 7 8 9 ] , 200 )
%      输出：
%          ans = 
%
%              [785x10 double]    [2.47962147181502]    [6]    [0.9231]
%
%      在这个对0-9个数字进行分类的实例中，训练得到的模型参数W是785x10的矩阵
%      W的二范数约是2.47962147181502，进行了6次迭代，对于测试数据的预测正确
%      率为92.31%。此外，程序运行耗时约40分钟。

%加载训练数据和测试数据
[ train , test ] = load_mnist( digits );

%将训练数据中所有x加入常量元素1作为截距
%将训练数据的输入矩阵X转置为每行一个样本，每列为一个维数
train.X = [ ones( 1 , length(train.y) ) ; train.X ]';

%将训练数据的标签y进行"one-of-K"编码，编码依据是digits向量
train.y = one_of_K( train.y , digits )';

%通过训练数据的输入矩阵和相对应的标签矩阵，在给定的正则化系数下
%求模型参数矩阵W，并输出迭代次数iterations
[ W , iterations ] = softmax_train( train.X , train.y , lambda );
%释放内存
clear train;

%将测试数据中所有x加入常量元素1作为截距
%将测试数据的输入矩阵X转置为每行一个样本，每列为一个维数
test.X = [ ones( 1 , length(test.y) ) ; test.X ]';

%将训练数据的标签y进行"one-of-K"编码，编码依据是digits向量
test.y = one_of_K( test.y , digits )';

%利用模型参数矩阵W，对测试数据集合的矩阵估计其相应标签集合矩阵

hat_T = softmax_test( test.X , W );

%真实标签矩阵test_y和预测标签矩阵y_hat，两者相减，除以两倍的样本数，得到错误次数
n = size( test.y , 1 );
accuracy = 1 - sum(sum(abs( test.y - hat_T ))) / ( n * 2 );

%输出 分类器参数矩阵W ， 参数W的二范数 ， 训练器迭代次数 ， 测试正确率
result={ W , norm(W) , iterations , accuracy };

end

