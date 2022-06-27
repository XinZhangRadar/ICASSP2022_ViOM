function [path,G_dist,W2] = compute_G_distance(X)
[m,n]=size(X);
D=zeros(m,m);
for i=1:m
    for j=i:m
        D(i,j)=norm(X(i,:)-X(j,:));
        D(j,i)=D(i,j);
    end
end
%计算矩阵中每行前k个值的位置并赋值（先按大小排列）
W1=zeros(m,m);
W2=zeros(m,m); 
k=round(0.8*m);
for i=1:m
A=D(i,:);%提出每一行
t=sort(A(:));%对每行进行排序后构成一个从小到大有序的列向量
[row,col]=find(A<=t(k),k);%找出每行前K个最小数的位置
for j=1:k
c=col(1,j);
 W1(i,c)=D(i,c); %W1(i,c)=1;%给k近邻赋值为距离
 W2(i,c)=D(i,c);
end
end
for i=1:m
    for j=1:m
        if W1(i,j)==0&&i~=j
            W1(i,j)=inf;
        end
    end
end
[path,G_dist] = compute_path(W1);