function Y = NN_test( X , Wb , activations )

W = Wb{ 1 };                                  %��ϵ������Wb��ȡ��W����

b = Wb{ 2 };                                  %��ϵ������Wb��ȡ��b����

activations = [ {[]} , activations ];         %�����û�м�����������Ԫ��ռλ

L = length( W );                              %��ȡ���������Ĳ���L������L+1��

A = X;                                        %ȡ��һ��ĵ�Ԫ������� A=X

for l = 1 : L                                 %ǰ�򴫲�
    
    Z = bsxfun( @plus , A * W{ l } , b{ l }); %�� l ��ĵ�Ԫ������󴫲����� l+1 ��
    
    A = activations{ l + 1 }( Z );            %�õ� l+1 ��ļ�������� �� l+1 ��ĵ�Ԫ�������
        
end

Y = A;                                        %���һ��ĵ�Ԫ���������������������

end

