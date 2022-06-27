
clc;
clear all;
close all;
%用mds对瑞士卷降维
 
%瑞士卷的生成图
N=1000;
t=(3*pi/2)*(1+2*rand(1,N));
s=21*rand(1,N);
X=[t.*cos(t);s;t.*sin(t)];
plot3(X(1,:),X(2,:),X(3,:),'.')
%计算距离矩阵个
X=X';
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
k=8;
for i=1:m
A=D(i,:);%提出每一行
t=sort(A(:));%对每行进行排序后构成一个从小到大有序的列向量
[row,col]=find(A<=t(k),k);%找出每行前K个最小数的位置
for j=1:k
c=col(1,j);
 W1(i,c)=D(i,c); %W1(i,c)=1;%给k近邻赋值为距离
end
end
for i=1:m
    for j=1:m
        if W1(i,j)==0&&i~=j
            W1(i,j)=inf;
        end
    end
end
%计算测地距离，o是每个点到其他点的测地距离矩阵
[path,o] = compute_path(W1);
[dist,mypath]=floyd(path,o,7,1000);
%dist
%mypath
 
[col,rol]=size(mypath);
X1=[];
for i=1:rol
    ding=mypath(1,i);
    X1=[X1;X(ding,:)];
end
plot3(X(:,1),X(:,2),X(:,3),'.')
hold on
plot3(X1(:,1),X1(:,2),X1(:,3),'o-r')
