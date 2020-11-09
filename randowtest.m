function randowtest
for i=1:10
    accu(i)=my_facerecognition1_2;
end
accu_mean=mean(accu)
a=0;
for i=1:10
    a=a+(accu(i)-accu_mean).^2;
end
a=a.^0.5
end