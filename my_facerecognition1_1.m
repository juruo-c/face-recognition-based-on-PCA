function my_facerecognition1_1
clear
close all
high=50;
wide=40;
k_all=40;%所有人类别数
k_train=9;%每一类的训练集个数
k_test=1;%每一类的测试集个数
X_train=[];%存放训练图像的矩阵
label_train=[];%存放训练集的类别
%读入训练数据
cnt=1;
for i=1:k_all
    for j=1:k_train
        if(i<10)
           a=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject0',num2str(i),'_',num2str(j),'.bmp'));     
        else
           a=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject',num2str(i),'_',num2str(j),'.bmp'));  
        end 
        b=a(1:(high*wide));
        b=double(b);
        X_train=[X_train;b];
        label_train(1,cnt)=i;
        cnt=cnt+1;
    end
end
%中心化训练样本
%1.计算平均脸(也就是计算训练矩阵每一行的均值并存于一个列向量中）
train_mean=mean(X_train);
%展示平均脸
imshow(reshape(train_mean,high,wide));
%2.将平均脸拓展为矩阵，并中心化训练样本
train_mean=repmat(train_mean,k_train*k_all,1);
%3.中心化操作
A=X_train-train_mean;

%PCA核心
%首先计算特征向量矩阵与特征值矩阵
S=A*A';%协方差矩阵
[V,D]=eig(S);
d1=diag(D);
%对特征值进行降序排序
[D_sort,index]=sort(d1,'descend');
D_sort=diag(D_sort,0);
%选择98%的能量
%显示能量占比图
[m,~]=size(D_sort);
dsum=trace(D);
dsort=reshape(diag(D_sort),1,m);
x=1:1:k_all*k_train;
p=0;
for i=1:1:(k_all*k_train)
    y(i)=sum(dsort(x(1:i)));
    %记录占98%能量的特征值个数
    if(y(i)/dsum<0.98)
        p=p+1;
    end
end
figure
y1=ones(1,k_all*k_train);
plot(x,y/dsum,x,y1*0.98,'linewidth',2);
grid
title('前n个特征值占总能量百分比');
xlabel('前n个特征值');
ylabel('所占百分比');
%对特征向量进行排序
for i=1:(k_all*k_train)
    V_sort(:,i)=V(:,index(i));
    d_sort(i)=d1(index(i));
end
%取出前p个特征向量组成一个特征矩阵
V_end=A'*V_sort;%恢复为原来的特征矩阵
i=1;
while (i<=p && d_sort(i)>0)      
    V_done(:,i) =V_end(:,i)*d_sort(i).^(-1/2);  %对人脸图像的标准化，特征脸
      i = i + 1; 
end 
%训练结束
%训练结果为特征矩阵V_done
%测试过程
Y_train=X_train*V_done;%对训练矩阵进行投影（也就是降维）
X_test=[];%存放测试图像数据
label_test=[];%存放测试图像标签
%读入测试数据
cnt=1;
for i=1:k_all
    for j=(k_train+1):(k_test+k_train)
        if(i<10)
           a=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject0',num2str(i),'_',num2str(j),'.bmp'));     
        else
           a=imread(strcat('C:\Users\19116\Desktop\PCA face recognition\image\subject',num2str(i),'_',num2str(j),'.bmp'));  
        end 
        b=a(1:(high*wide));
        b=double(b);
        X_test=[X_test; b];
        label_test(1,cnt)=i;
        cnt=cnt+1;
    end
end
%对测试数据投影到特征空间（降维）
Y_test=X_test*V_done;
%KNN分类器识别Y_test
class_test=[];%存放测试结果
% k=5;%knn的k
for i=1:k_all*k_test
    dis=[];%存放该测试样本与每一个训练样本的距离
    for j=1:k_all*k_train
        %计算欧几里得距离
        distance=0;
        for m=1:p
            distance=distance+(Y_train(j,m)-Y_test(i,m)).^2;
        end
        dis(1,j)=distance.^0.5;
    end
    %找到3个最近邻
    [~,index]=sort(dis);
    class1=label_train(index(1));      
    class2=label_train(index(2));        
    class3=label_train(index(3));        
    if class1~=class2 && class2~=class3 
            class=class1;         
        elseif class1==class2          
            class=class1;         
        elseif class2==class3     
            class=class2;         
    end
        class_test(i)=class;
end
% %计算识别率
accu=0;%统计正确数量
for i=1:k_all*k_test
    if class_test(i)==label_test(i)
        accu=accu+1;
    end
end
accuracy=accu/(k_all*k_test)
end