function Result = main
% 直接运行main即可
% 
% 在本程序可控制：隐藏层数、各层单元数、各层激活函数，梯度下降步长，程序迭代次数
% 
% 程序将输出迭代最终结果的参数集合 W, b，以及相应的测试正确率
% 
% 当使用一个100单元的隐藏层，隐藏层激活函数用sigmoid，输出层激活函数用softmax，用步长2进行迭代，迭代1000次的正确率为0.962，迭代1500次正确率为0.9648，迭代2000次正确率为0.9666。
%
% 当使用一个100单元的隐藏层，隐藏层激活函数用tanh，输出层激活函数用softmax，用步长2进行迭代，迭代200次的正确率为0.9496，迭代1000次正确率为   ，迭代2000次正确率为   。
%
% 若步长设置太大会发生震荡，太小则收敛很慢
%
% 可以尝试用不同的层数或者不同的单元数
% 
% 也可以另外调用或者编写不同的激活函数来使用，但是要对隐藏层给出相应的导函数
%
% 有建议或者发现漏洞请联系matthew

%设置所调用激活函数及导函数的路径
addpath Activations_functions

%设置分类目标的数字
Digits = [ 0 1 2 3 4 5 6 7 8 9 ];

%设置网络结构，Config的长度控制层数，各个值控制各层的单元数量。首值和末值要分别与输入向量、输出向量维数一致
Config = [ 784 100 10 ];

%设置隐藏层和输出层的激活函数，每个隐藏层和输出层都要设置一个激活函数
Activations = { @tanh , @softmax };
%Activations = { @sigmoid , @softmax };

%设置与隐藏成激活函数对应的导函数，每个隐藏层都要设置一个激活函数
Dervatives = { @diff_tanh };
%Dervatives = { @diff_sigmoid };

%设置迭代次数，本程序通过最大迭代次数来控制停止
Max_iterations = 200;

%设置步长
Eta = 1.5;

%利用给定参数运行NN，并将结果作为Result输出
Result = mnist_NN( Digits, Config, Activations, Dervatives, Max_iterations, Eta );

end