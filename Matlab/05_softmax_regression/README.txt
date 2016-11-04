主调程序，用于准备数据，训练模型参数，用模型对测试数据进行分类估计并计算正确率
result = softmax_mnist( digits , lambda )

训练模块，用给定的训练集合来训练模型的参数
[ W , iterations ] = softmax_train( X , T , lambda )

测试模块，用训练得到的模型参数来对测试数据分类估计
T = softmax_test( X , W )

softmax变换函数，用模型参数，求输入向量或输入向量集合矩阵所属于各类的概率
Y = softmax_hypothesis_function( X , W , is_column )

"one-of-K"编码函数，对elements的每个元素，依据目录向量target进行编码
labels = one_of_K( elements , targets )

加载数据模块，用于加载训练数据集合和测试数据集合
[ train, test ] = load_mnist( digits )

在构造Hessian矩阵的时候，需要做大量矩阵乘法运算，耗费大量时间，因此在调试阶段可以减少类别数


示例：
    
          >> softmax_mnist( [ 0 1 2 3 4 5 6 7 8 9 ] , 200 )

          ans = 

              [785x10 double]    [2.47962147181502]    [6]    [0.9231]

%      在这个对0-9个数字进行分类的实例中，训练得到的模型参数W是785x10的矩阵
%      W的二范数约是2.47962147181502，进行了6次迭代，对测试数据的预测正确率
%      为92.31%。此外，本程序运行耗时约30分钟。