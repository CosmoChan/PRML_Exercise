function Y = NN_test( X , Wb , activations )

W = Wb{ 1 };                                  %��ϵ������Wb��ȡ��W����

b = Wb{ 2 };                                  %��ϵ������Wb��ȡ��b����

activations = [ {[]} , activations ];         %�����û�м�����������Ԫ��ռλ

L = length( W );                              %��ȡ���������Ĳ���L������L+1��

Z = X;                                        %ȡ��һ��ĵ�Ԫ������� Z=X

for l = 1 : L                                 %ǰ�򴫲�
    
    A = bsxfun( @plus , Z * W{ l } , b{ l }); %�� l ��ĵ�Ԫ������󴫲����� l+1 ��
    
    Z = activations{ l + 1 }( A );            %�õ� l+1 ��ļ�������� �� l+1 ��ĵ�Ԫ�������
        
end

Y = Z;                                        %���һ��ĵ�Ԫ���������������������

end

