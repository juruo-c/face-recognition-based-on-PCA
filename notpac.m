function notpac
high=100;
wide=80;
k_all=15;%所有人类别数
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
        b=reshape(a,high*wide,1);
        b=double(b);
        X_train=[X_train b];
        label_train(1,cnt)=i;
        cnt=cnt+1;
    end
end
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
        b=reshape(a,high*wide,1);
        b=double(b);
        X_test=[X_test b];
        label_test(1,cnt)=i;
        cnt=cnt+1;
    end
end
%三近邻
class_test=[];%存放测试结果
for i=1:k_all*k_test
    dis=[];%存放该测试样本与每一个训练样本的距离
    for j=1:k_all*k_train
        %计算欧几里得距离
        distance=0;
        for m=1:(high*wide)
            distance=distance+(X_train(m,j)-X_test(m,i)).^2;
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