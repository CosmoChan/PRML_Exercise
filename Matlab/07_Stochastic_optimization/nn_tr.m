function arg = nn_tr( train, alpha, iteration )
%nn_tr ѵ��������ģ��
%   train ѵ������
%   alpha ������ѧϰ����
%   iteration ����������������
%   W_1,W_2,b_1,b_2 ������Ĳ���

[n,m] = size(train.X);

% ����������Ĳ���
arg.W_1 = 0.05 * randn(100, 784);
arg.W_2 = 0.05 * randn(10, 100);
arg.b_1 = 0.05 * randn(100, 1);
arg.b_2 = 0.05 * randn(10, 1);

% ʹ��ѭ�������Ż����������
for i=1:iteration
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %���ѡ��ѵ������X���Լ���Ӧ�ر�ǩY







    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    a_1 = X;%784*60000
    
    % �������ز�������Ȩ��z_2
    z_2 = arg.W_1 * a_1 + arg.b_1;%100*60000
    % �������ز�ļ���ֵa_2
    a_2 = Sigmoid(z_2);%100*60000
    
    % ���������������Ȩ��z_3
    z_3 = arg.W_2 * a_2 + arg.b_2;%10*60000
    % ���������ļ���ֵa_3
    a_3 = softmax(z_3);%10*60000
    
    % ������ۺ���J
    J = 1/(2 * m) * sum(sum((a_3 - Y).*(a_3 - Y)));
    
    % ���з��򴫲�
    % ���������Ĳв�
    delta_3 = (a_3 - Y);% 10*60000
    % �������ز�Ĳв�
    delta_2 = (arg.W_2' * delta_3) .* a_2 .* (1 - a_2);% 100*60000
    
    % �����ݶ�
    Delta_W_2 = delta_3 * a_2';%10*100
    Delta_b_2 = sum(delta_3,2);%10*1
    Delta_W_1 = delta_2 * a_1';%100*784
    Delta_b_1 = sum(delta_2,2);%100*1
    
    % ����ϵ��
    arg.W_1 = arg.W_1 - alpha * 1/m * Delta_W_1;
    arg.W_2 = arg.W_2 - alpha * 1/m * Delta_W_2;
    arg.b_1 = arg.b_1 - alpha * 1/m * Delta_b_1;
    arg.b_2 = arg.b_2 - alpha * 1/m * Delta_b_2;
    
end
end