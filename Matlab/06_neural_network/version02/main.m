function Result = main
% 直接运行main即可
% 
%在本程序中可调节层数、各层单元数、各层激活函数，梯度下降步长，程序迭代次数
% 
% 程序将输出参数W集合和b集合
% 
% 当使用一个100单元的隐藏层，隐藏层激活函数用sigmoid，输出层激活函数用softmax，用步长2进行迭代，迭代1000次的正确率为0.962，迭代1500次正确率为0.9648，迭代2000次正确率为0.9666。若步长设置太大会发生震荡，太小则收敛很慢
% 
% 发现漏洞或者有建议请联系matthew

%设置隐藏层和输出层的激活函数
Activations = { @sigmoid , @softmax };

%设置隐藏层
Dervatives = { @diff_sigmoid };

%设置分类目标的数字
Digits = [ 0 1 2 3 4 5 6 7 8 9 ];

%设置3层单元，输入层784个单元，隐藏层800个单元，输出层10个单元
Config = [ 784 100 10 ];

%设置迭代次数，本程序通过最大迭代次数来控制停止
Max_iterations = 1000;

%设置步长
Eta = 2;

%输出结果
Result = mnist_NN( Digits, Config, Activations, Dervatives, Max_iterations, Eta );

end