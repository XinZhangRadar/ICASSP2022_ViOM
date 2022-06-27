function [path,o] = compute_path(a)
n=size(a,1); path=zeros(n);
for i=1:n
for j=1:n
if a(i,j)~=inf
path(i,j)=j; %j 是i 的后续点
end
end
end
for k=1:n
for i=1:n
for j=1:n
if a(i,j)>a(i,k)+a(k,j)
a(i,j)=a(i,k)+a(k,j);
path(i,j)=path(i,k);
end
end
end
end
o = a;
