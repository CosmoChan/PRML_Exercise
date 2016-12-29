function image = image_segmentation( image , K , alpha , max_iterations )
% image_segmentation.m
%     ����K-means�����㷨��ͼ��ָ���� wuweizhen version
% ����
%     image (height,width,d)��ά���� ͼ��߶�Ϊheight�����Ϊwidth����ɫͨ����Ϊd
%     K     ͼ�����ķָ���Ŀ
%     alpha λ����Ϣ��Ȩ��ϵ����alphaԽ��λ����Ϣ�Էָ���Ӱ��Խ�󣬵�alpha=0ʱ�ָ����λ����Ϣ
%     max_iterations  ����������
% ���
%     image (height,width,d)��ά���飬�����ָ���ͼ��

[ height , width , d ] = size( image );                    %��ȡͼ��߶ȡ���ȡ���ɫͨ����

[ x , y ] = meshgrid( 1 : height , 1 : width );            %����һ�����񣬸�ͼ��ÿ����������һ������(x,y)

X = double( image );                                       %��ͼ����uint8��ʽת��Ϊdouble��ʽ

X( : , : , d+1 ) = x;                                      %��ͼ����ά����ĵ�d+1ҳ����x����
    
X( : , : , d+2 ) = y;                                      %��ͼ����ά����ĵ�d+2ҳ����y����

X = reshape( X ,  height * width , d+2 );                  %��ͼ������ά�������Ϊ����ÿ����һ���۲�ֵ��ÿ����һ��ά

Std = std( X , [] , 1 ) + eps;                             %����ÿ��ά�ı�׼��

Std( d+1 : d+2 ) = Std( d+1 : d+2 ) / (alpha+eps);         %��λ�õ�ά������Ȩ��ϵ��alpha

X = bsxfun( @rdivide , X , Std );                          %��һ��

[ labels , means , ~ ] = K_means( X , K , max_iterations );%����K-means����

for k = 1 : K
                                                           %��ÿһ����������ɫ���ø����ƽ����ɫ����ֵ
    X( labels==k , : ) = repmat( means( k , : ) , sum( labels == k ) , 1 );
    
end

X = bsxfun( @times , X , Std );                             %�����ݽ���Ԥ�������任���任��ԭ�ռ�

X = reshape( X( : , 1 : d ) , height , width , d );         %ֻȡ����ɫ��Ϣ�����±���Ϊ(height,width,d)����ά������ʽ

image = uint8( X );                                         %������ת��Ϊuint8

end