function [dist,mypath]=floyd(path,a,sb,db)
% 输入：a—邻接矩阵(aij)是指i 到j 之间的距离，可以是有向的
% sb—起点的标号；db—终点的标号
% 输出：dist—最短路的距离；% mypath—最短路的路径

dist=a(sb,db);
mypath=sb; t=sb;
while t~=db
temp=path(t,db);
mypath=[mypath,temp];
t=temp;
end
return
